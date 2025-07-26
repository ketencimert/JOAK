# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 23:22:45 2025

@author: Mert
"""

from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from pmi_utils.gmm_pmi_utils import GMM
from pmi_utils.shared_pmi_utils import generate_combinations, \
    simple_dataset_epoch

class GaussianMixturePMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            max_interaction_depth: int = 2,
            n_components: int = 100,
            batch_size: int = 1024,
            epochs: int = 3000,
            max_to_keep: int = 1,
            ):
        super().__init__()
        """
        📌 X: Dataset to model PMI -> N by D numpy array

        📌 max_interaction_depth: How many interactions we will model
        -> integer

        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_interaction_depth = max_interaction_depth
        self.eps = 1e-3

        input_size = X.shape[-1]
        self.interactions = generate_combinations(
            range(input_size), max_interaction_depth
            )
        
        self.estimator = GMM(
            n_components=n_components,
            n_dims=input_size
            )

        X = X.astype(np.float32)
        D = X.shape[1]
        X_train, X_val, _, _ = train_test_split(
            X, X, test_size=0.33, random_state=42
            )

        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: simple_dataset_epoch(X_train, batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, D), dtype=tf.float32)
                )
            ).prefetch(tf.data.AUTOTUNE)
        self.total = sum(1 for _ in self.train_dataset)

        self.valid_dataset = tf.data.Dataset.from_generator(
            lambda: simple_dataset_epoch(X_val, batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, D), dtype=tf.float32)
                )
            ).prefetch(tf.data.AUTOTUNE)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3, weight_decay=1e-4
            )
        
        self.filename = datetime.now().strftime("pmi_model_%Y%m%d_%H%M%S")
        self.best_val_loss = tf.Variable(float('inf'), trainable=False)
        self.checkpoint = tf.train.Checkpoint(
            model=self.estimator,
            optimizer=self.optimizer,
            best_val_loss=self.best_val_loss
            )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.filename, max_to_keep=max_to_keep
            )

    def train(self):
        """
        📌 Learn GMM to estimate the data density.
        """
        try:
            print('Fitting PMI estimation module.')
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                for step, x_batch in enumerate(
                        tqdm(
                            self.train_dataset,
                            desc=f"Epoch {epoch+1}",
                            total=self.total
                            )
                        ):
                    with tf.GradientTape() as tape:
                        loss =-tf.reduce_mean(
                            self.estimator.log_prob(x_batch)
                            ) 

                    grads = tape.gradient(
                        loss, self.estimator.trainable_variables
                        )
                    self.optimizer.apply_gradients(
                        zip(grads, self.estimator.trainable_variables)
                        )

                avg_val_loss = self.evaluate_model_on_validation_dataset()
                print(
                    f"Epoch {epoch+1}: val_loss = {avg_val_loss:.4f}"
                    f"  | best = {self.best_val_loss.numpy():.4f}"
                    )
                # 💾 Save if improved
                cond1 = avg_val_loss <= self.best_val_loss
                cond2 = np.isclose(
                    avg_val_loss, self.best_val_loss.numpy(), atol=2e-5
                    )
                # Validation could be good/bad due to noise. So, let's save
                # if it is close to the best score with 2e-5 dist.
                if cond1 or cond2:
                    print("✅ New potential best model, saving checkpoint...")
                    self.manager.save()
                    if cond1:
                        self.best_val_loss.assign(avg_val_loss)
                        print(f"✅ New best loss: {avg_val_loss:.4f}")
                else:
                    print(
                        f"Loss: {avg_val_loss:.4f} | "
                        f"Best: {self.best_val_loss.numpy():.4f}"
                        )
            print('Training complete. Loading weights.')
            self.load_top_model()
        except KeyboardInterrupt:
            print('KeyboardInterrupt. Loading weights and ending training.')
            self.load_top_model()

    def evaluate_model_on_validation_dataset(self):
        val_losses = []
        for x_val in self.valid_dataset:
            val_loss =-tf.reduce_mean(
                self.estimator.log_prob(x_val)
                ) 
            val_losses.append(val_loss.numpy())

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    def load_top_model(self):
        # Get all saved checkpoints
        saved_checkpoints = self.manager.checkpoints
        checkpoint_losses = []
        for path in tqdm(saved_checkpoints, total=len(saved_checkpoints)):
            self.checkpoint.restore(path).expect_partial()
            print(f"Evaluating {path}")
            avg_loss = self.evaluate_model_on_validation_dataset()
            checkpoint_losses.append((path, avg_loss))
        # Pick the best
        best_ckpt_path, best_loss = min(checkpoint_losses, key=lambda x: x[1])
        print(
            f"\nBest checkpoint: {best_ckpt_path}"
            f"with avg loss: {best_loss:.4f}"
            )
        self.checkpoint.restore(best_ckpt_path).expect_partial()

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
            #returns logits: log (p/1-p)
            if len(interaction) <= 1:
                inv_exp_pmi_dict[tuple(interaction)] \
                    = tf.convert_to_tensor(np.ones((x.shape[0], 1)))
            else:
                #marginal logprob
                exp_pmi_vals1 = tf.exp(tf.reduce_sum(tf.stack([
                    self.estimator.marginal_log_prob(x, [i])
                    for i in interaction
                    ], -1), -1))
                #joint logprob
                exp_pmi_vals2 = tf.exp(
                    self.estimator.marginal_log_prob(x, interaction)
                    )
                #marginal / joint are kernel weights =
                #inv exp pmi values
                inv_exp_pmi_vals = exp_pmi_vals1 / (exp_pmi_vals2 + self.eps)
                # inv_exp_pmi_vals = inv_exp_pmi_vals ** (-1)
                inv_exp_pmi_dict[tuple(interaction)] = tf.stop_gradient(
                    inv_exp_pmi_vals
                    )
        return inv_exp_pmi_dict

if __name__ == '__main__':

    # Test runs
    # Create Two Moons dataset
    X, _ = make_moons(n_samples=30000, noise=0.1)
    # X = np.random.normal(0, 1, size=(30000,2))
    N, D = X.shape
    X = X.astype(np.float32)

    # Define the model
    model = GaussianMixturePMIModel(
        X=X,
        )
    model.train()

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack(
        [grid_x0.ravel(), grid_x1.ravel()], axis=1
        ).astype(np.float32)
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

