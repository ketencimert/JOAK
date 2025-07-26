from itertools import combinations

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
import tensorflow_probability as tfp

tfd = tfp.distributions

def generate_combinations(lst, C):
    """
    📌 Generate all possible combinations of the elements in a list.

    📌 lst: List of indices. e.g. [0,1,2]

    📌 C: Maximum order of combinations

    📌 e.g., if input is [0,1,2] output will be
    [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
    """
    all_combos = []
    for r in range(1, C + 1):
        all_combos.extend(combinations(lst, r))
    return [list(c) for c in all_combos]

class GaussianPMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            max_interaction_depth=2,
            ):
        super().__init__()
        """
        📌 X: Dataset to model PMI -> N by D numpy array

        📌 max_interaction_depth: How many interactions we will model
        -> integer
        """

        input_size = X.shape[-1]
        self.interactions = generate_combinations(
            range(input_size), max_interaction_depth
            )

    def train(self, X):
        """
        📌 Estimate the mean vector and covariance matrix of a dataset 
        using TensorFlow.
        
        📌 Args:
            📌 X: tf.Tensor of shape (N, D), where N is the number of samples 
            and D is the number of dimensions.
    
        📌 Returns:
             📌 mean: tf.Tensor of shape (D,)
             📌 covariance: tf.Tensor of shape (D, D)
        """
        # Convert to float32 if necessary
        X = tf.cast(X, tf.float32)
    
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

        📌 input: typically the values you will feed into your OAK-Kernel

        📌 output: exponential pmi values given an input instance x.
        if x is 3 dimensional and your are modeling all interactions than
        columns will represent:
        [null, 0, 1, 2, (0,1), (0,2), (1,2), (1,2,3)]

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
                inv_exp_pmi_dict[tuple(interaction)] = inv_exp_pmi_vals

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
    model.train(X)

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack([grid_x0.ravel(), grid_x1.ravel()], axis=1)
    pmi_points = model.inv_exp_pmi_array_(grid_points) #for testing
    for points in pmi_points.T:
        # Predict probabilities and compute inverse PMI
        exp_pmi_vals = points
        pmi_grid = exp_pmi_vals.reshape(grid_x0.shape)
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
