# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 23:22:45 2025

@author: Mert
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import make_moons

tfd = tfp.distributions
tfb = tfp.bijectors

# Define GMM Model
class GMM(tf.keras.Model):
    def __init__(self, n_components=5, n_dims=2):
        super().__init__()
        self.n_components = n_components
        self.n_dims = n_dims

        self.logits = tf.Variable(
            tf.zeros([n_components]), name="logits"
            )
        self.locs = tf.Variable(
            tf.random.normal([n_components, n_dims]), name="locs"
            )
        self.scales = tfp.util.TransformedVariable(
            tf.ones([n_components, n_dims]), 
            bijector=tfb.Softplus(), 
            name="scales"
            )

    def distribution(self):
        components = tfd.MultivariateNormalDiag(
            loc=self.locs, scale_diag=self.scales
            )
        mixture = tfd.Categorical(logits=self.logits)
        return tfd.MixtureSameFamily(
            mixture_distribution=mixture, 
            components_distribution=components
            )
    
    def marginal_distribution(self, indices):
        locs =  tf.gather(self.locs, indices, axis=-1)
        scales = tf.gather(self.scales, indices, axis=-1)
        components = tfd.MultivariateNormalDiag(
            loc=locs, scale_diag=scales
            )
        mixture = tfd.Categorical(logits=self.logits)
        return tfd.MixtureSameFamily(
            mixture_distribution=mixture, 
            components_distribution=components
            )
    
    def marginal_log_prob(self, x, indices):
        x = tf.gather(x, indices, axis=-1)
        return self.marginal_distribution(indices).log_prob(x)
    
    def log_prob(self, x):
        return self.distribution().log_prob(x)

    def sample(self, n):
        return self.distribution().sample(n)

if __name__ == '__main__':
    
    # TEST YOUR GAUSSIAN MIXTURE MODEL FOR DENSITY ESTIMATION!
    
    # Train the GMM
    model = GMM(n_components=500, n_dims=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    # Generate toy 2D data (two moons)
    X, y = make_moons(n_samples=500, noise=0.05)
    X = X.astype(np.float32)

    for step in range(3000):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(model.log_prob(X))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Get trained distribution
    gmm = model.distribution()
    
    # Create grid for density
    xlin = np.linspace(-2, 3, 200)
    ylin = np.linspace(-1.5, 2, 200)
    xx, yy = np.meshgrid(xlin, ylin)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    # Evaluate GMM density
    log_probs = gmm.log_prob(grid).numpy()
    probs = np.exp(log_probs).reshape(xx.shape)
    
    # Sample from the GMM
    samples = gmm.sample(500).numpy()
    
    # Plot density and samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].contourf(xx, yy, probs, levels=100, cmap="viridis")
    # axes[0].scatter(X[:, 0], X[:, 1], c='white', s=10, alpha=0.6)
    axes[0].set_title("GMM Density with Original Data")
    axes[0].axis("equal")
    
    axes[1].scatter(
        samples[:, 0], samples[:, 1], c='tomato', s=10, alpha=0.7
        )
    axes[1].set_title("Samples from Trained GMM")
    axes[1].axis("equal")
    
    plt.tight_layout()
    plt.show()