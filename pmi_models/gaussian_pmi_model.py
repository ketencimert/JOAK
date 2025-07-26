# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 23:22:45 2025

@author: Mert
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
import tensorflow_probability as tfp

from pmi_utils.shared_pmi_utils import generate_combinations

tfd = tfp.distributions

class GaussianPMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            max_interaction_depth: int = 2,
            ):
        super().__init__()
        """
        📌 A single Gaussian to estimate kernel weights.
        📌 Args:
            📌 X: Dataset to model PMI -> N by D numpy array
            📌 max_interaction_depth: How many interactions we will model
        """

        input_size = X.shape[-1]
        self.interactions = generate_combinations(
            range(input_size), max_interaction_depth
            )
        self.X = X

    def train(self):
        """
        📌 Learn a single Gaussian to estimate the data density.
        """
        # Convert to float32 if necessary
        X = tf.cast(self.X, tf.float32)
    
        # Compute mean
        self.mean = tf.reduce_mean(X, axis=0)  # shape (D,)
    
        # Center the data
        X_centered = X - self.mean  # shape (N, D)
    
        # Compute covariance matrix: (X^T X) / (N - 1)
        N = tf.cast(tf.shape(X)[0], tf.float32)
        self.covariance = tf.matmul(
            X_centered, X_centered, transpose_a=True
            ) / (N - 1.0)  # shape (D, D)

    def inv_exp_pmi_dict(self, x):
        """
        For JOAK, use this.
        📌 Args:
            📌 input: typically the values you will feed into your OAK-Kernel

        📌 Returns:
            📌 inv_exp_pmi_dict: 
                exponential inverse pmi values given an input instance x.
                keys are interaction values (tuple) and values are tf arrays.
            📌 i.e., N by 2^D PMI values (per-instance by per-interaction)
        """
        inv_exp_pmi_dict = dict()
        for interaction in [[]] + self.interactions:
            m = np.zeros(x.shape)
            m[:,interaction] = 1
            #returns logits: log (p/1-p)
            if len(interaction) <= 1:
                inv_exp_pmi_dict[tuple(interaction)] \
                    = tf.convert_to_tensor(np.ones((x.shape[0], 1)))
            else:
                x_ = x[:,interaction]
                covariance = tf.gather(
                    tf.gather(
                        self.covariance, interaction, axis=0
                        ), interaction, axis=1
                    )
                covariance_diag = tf.linalg.diag_part(covariance)
                mean = tf.gather(
                        self.mean, interaction, axis=0
                        )
                inv_pmi_vals = tfd.MultivariateNormalDiag(
                    loc=mean,
                    scale_diag=covariance_diag ** 0.5
                    ).log_prob(x_) - tfd.MultivariateNormalFullCovariance(
                    loc=mean,
                    covariance_matrix=covariance
                    ).log_prob(x_)
                inv_exp_pmi_vals = tf.exp(inv_pmi_vals)
                inv_exp_pmi_vals = tf.reshape(inv_exp_pmi_vals, (-1,1))
                inv_exp_pmi_dict[tuple(interaction)] = tf.stop_gradient(
                    inv_exp_pmi_vals
                    )

        return inv_exp_pmi_dict


if __name__ == '__main__':

    # Test runs
    # Create Two Moons dataset
    X, _ = make_moons(n_samples=30000, noise=0.1)
    # X = np.random.normal(0, 1, size=(30000,2))
    
    # Mean vector (D = 2)
    # mean = np.array([0.0, 0.0])
    
    # # Covariance matrix with correlation (ρ = 0.8)
    # cov = np.array([[1.0, 0.999],
    #                 [0.999, 1.0]])
    
    # Generate N samples
    # N = 30000
    # X = np.random.multivariate_normal(mean, cov, size=N)

    N, D = X.shape
    X = X.astype(np.float32)

    # Define the model
    model = GaussianPMIModel(
        X=X,
        )
    model.train()

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack([grid_x0.ravel(), grid_x1.ravel()], axis=1)
    pmi_points = model.inv_exp_pmi_dict(grid_points) #for testing
    for points in pmi_points.values():
        # Predict probabilities and compute inverse PMI
        exp_pmi_vals = points.numpy()
        pmi_grid = exp_pmi_vals.reshape(grid_x0.shape)
        # Plot the estimated log inverse PMI
        plt.figure(figsize=(6, 5))
        plt.contourf(
            grid_x0, grid_x1, pmi_grid,
            #norm=divnorm,
            levels=200, cmap='viridis'
            )
        plt.colorbar(label='Inverse EXP PMI Estimate')
        plt.xlabel('x₀')
        plt.ylabel('x₁')
        plt.tight_layout()
        plt.show()
