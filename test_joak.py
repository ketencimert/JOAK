
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from joak.model_utils import joak_model, save_model
from joak.utils import get_model_sufficient_statistics, get_prediction_component
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path
# +
# data from repo: https://github.com/duvenaud/additive-gps/blob/master/data/regression/
# this script is for experiments in Sec 5.1 for regression problems in the paper
data_path_prefix = os.path.join(
    Path(os.getcwd()).parent.parent.absolute(), f"data/"
)

np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})

dataset_name, k = "autoMPG", 5
filename = data_path_prefix + "autompg.mat"
covariate_names = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin",
]
output_prefix = 'test'
# save results to outputs folder
try:
    if not os.path.exists(f"./outputs/{dataset_name}/"):
        os.mkdir(output_prefix)
except FileExistsError:
    pass

np.random.seed(4)
tf.random.set_seed(4)

print(f"dataset {dataset_name}\n")

#Load data
d = io.loadmat("C:\\Users\\Mert\\Desktop\\research\\github\\idea 2 - OAK\\JOAK\\data\\autompg.mat")
if dataset_name == "autoMPG":
    # for autoMPG dataset, the first column is the response y
    X, y = d["X"][:, 1:], d["X"][:, :1]
else:
    X, y = d["X"], d["y"]

idx = np.random.permutation(range(X.shape[0]))

X = X[idx, :]
y = y[idx]
kf = KFold(n_splits=k)

# take one fold for example 
fold = 0
for train_index, test_index in kf.split(X):
    if fold == 0:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    fold += 1
  
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#This is where the test begins
###############################################################################
###############################################################################
###############################################################################
###############################################################################

joak = joak_model(max_interaction_depth=X.shape[1])
joak.fit(X_train, y_train)

# test performance
x_max, x_min = X_train.max(0), X_train.min(0)
y_pred = joak.predict(np.clip(X_test, x_min, x_max))
rss = ((y_pred - y_test[:, 0]) ** 2).mean()
tss = (
    (y_test[:, 0] - y_test[:, 0].mean() * np.ones(y_test[:, 0].shape)) ** 2
).mean()
r2 = 1 - rss / tss
rmse = np.sqrt(rss)

# calculate sobol for each term in the decomposition
joak.get_sobol()
tuple_of_indices, normalised_sobols = (
    joak.tuple_of_indices,
    joak.normalised_sobols,
)

# Get predictions for each term (kernel) in prediction_list
x_max, x_min = X_train.max(0), X_train.min(0)
XT = joak._transform_x(np.clip(X_test, x_min, x_max))
joak.alpha = get_model_sufficient_statistics(joak.m, get_L=False)
# get the predicted y for all the kernel components
prediction_list = get_prediction_component(
    joak.m,
    joak.alpha,
    XT,
)
# predicted y for the constant kernel
constant_term = joak.alpha.numpy().sum() * joak.m.kernel.variances[0].numpy()
print(f"constant_term = {constant_term}")
y_pred_component = np.ones(y_test.shape[0]) * constant_term

# get prediction performance and cumulative Sobol as we add terms one by one
# ranked by their Sobol (most important kernel first)
cumulative_sobol, rmse_component = [], []
order = np.argsort(normalised_sobols)[::-1]
for n in order:
    # add predictions of the terms one by one ranked by their Sobol index
    y_pred_component += prediction_list[n].numpy()
    y_pred_component_transformed = joak.scaler_y.inverse_transform(
        y_pred_component.reshape(-1, 1)
    )
    error_component = np.sqrt(
        ((y_pred_component_transformed - y_test) ** 2).mean()
    )
    rmse_component.append(error_component)
    cumulative_sobol.append(normalised_sobols[n])
cumulative_sobol = np.cumsum(cumulative_sobol)

# sanity check that predictions by summing over the components is equal
# to the prediction of the joak model
np.testing.assert_allclose(y_pred_component_transformed[:, 0], y_pred)

joak.plot(
    top_n=5,
    semilogy=False,
    X_columns=covariate_names,
)