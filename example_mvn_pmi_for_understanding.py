# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 21:43:48 2025

@author: Mert
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Example: Random 2D data
np.random.seed(0)
X = np.random.randn(2) * 10  # 100 samples, 2-dimensional

# Define mean and covariance of the MVN
mu = np.array([0.0, 0.0])

s11 = 1.0
s22 = 1.0
s12 = 0.5
s21 = s12

Sigma = np.array([[s11, s12],
                  [s21, s22]])

# Create MVN object
mvn = multivariate_normal(mean=mu, cov=Sigma)
univariate_normal1 = norm(loc=mu[0], scale=s11)
univariate_normal2 = norm(loc=mu[1], scale=s22)


# Compute log-likelihood for each point and sum
log_likelihood = univariate_normal1.logpdf(X[0]) \
    +  univariate_normal2.logpdf(X[1])
log_likelihood = log_likelihood - mvn.logpdf(X)
likelihood = np.exp(log_likelihood)
print(likelihood)
