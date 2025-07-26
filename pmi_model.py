# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:40:35 2025

@author: Mert
"""

from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

from pmi_models.gaussian_pmi_model import GaussianPMIModel
from pmi_models.gmm_pmi_model import GaussianMixturePMIModel
from pmi_models.neural_pmi_model import NeuralPMIModel

class PMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            pmi_model_type: str = 'neural',
            max_interaction_depth: int = 2,
            activation: str = 'elu',
            embedding_size: int = 20,
            masked_units: List[int] = [200, 200],
            hidden_units: List[int] = [64, 64, 64],
            batch_size: int = 1024,
            epochs: int = 1000,
            max_to_keep: int = 20,
            num_evaluation_trials: int = 25, #reduce the variance 5 times
            n_components: int = 100,
            ):
        super().__init__()

        if pmi_model_type == 'neural':
            self.model = NeuralPMIModel(
                X=X,
                max_interaction_depth=max_interaction_depth,
                activation=activation,
                embedding_size=embedding_size,
                masked_units=masked_units,
                hidden_units=hidden_units,
                batch_size=batch_size,
                epochs=epochs,
                max_to_keep=max_to_keep,
                num_evaluation_trials=num_evaluation_trials,
                )
        elif pmi_model_type == 'gmm':
            self.model = GaussianMixturePMIModel(
                X=X,
                max_interaction_depth=max_interaction_depth,
                n_components=n_components,
                batch_size=batch_size,
                epochs=epochs,
                max_to_keep=max_to_keep,
                )
        elif pmi_model_type == 'gaussian':
            self.model = GaussianPMIModel(
                X=X,
                max_interaction_depth=max_interaction_depth,
                )

    def train(self):
        self.model.train()
        self.model.trainable = False
        self.model._trainable = False

    def compute_kernel_weights(self, x):
        return self.model.inv_exp_pmi_dict(x)

if __name__ == '__main__':

    # Test runs
    # Create Two Moons dataset
    X, _ = make_moons(n_samples=30000, noise=0.1)
    # X = np.random.normal(0, 1, size=(30000,2))
    N, D = X.shape
    X = X.astype(np.float32)

    # Define the model
    model = PMIModel(
        X=X, 
        pmi_model_type='neural'
        )
    model.train()

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack([grid_x0.ravel(), grid_x1.ravel()], axis=1)
    
    pmi_points = model.compute_kernel_weights(grid_points) #for testing
    for points in pmi_points.values():
        # Predict probabilities and compute inverse PMI
        exp_pmi_vals = points.numpy()
        pmi_grid = np.log(exp_pmi_vals.reshape(grid_x0.shape))
        # Plot the estimated log inverse PMI
        plt.figure(figsize=(6, 5))
        plt.contourf(
            grid_x0, grid_x1, pmi_grid,
            #norm=divnorm, 
            levels=200, cmap='viridis'
            )
        plt.colorbar(label='PMI Estimate')
        plt.title(
            'Estimated PMI(x₀; x₁) via Discriminator (Shuffled In-Model)'
            )
        plt.xlabel('x₀')
        plt.ylabel('x₁')
        plt.tight_layout()
        plt.show()