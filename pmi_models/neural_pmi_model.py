# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 21:27:35 2025

@author: Mert
"""

from typing import List

from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from pmi_utils.neural_pmi_utils import PMINetwork
from pmi_utils.shared_pmi_utils import generate_combinations,\
    uniform_pmi_dataset_epoch, shapley_pmi_dataset_epoch

class NeuralPMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            max_interaction_depth: int = 2,
            activation: str = 'elu',
            embedding_size: int = 20,
            masked_units: List[int] = [200, 200],
            hidden_units: List[int] = [64, 64, 64],
            batch_size: int = 1024,
            epochs: int = 3000,
            max_to_keep: int = 20,
            num_evaluation_trials: int = 25, #reduce the variance 5 times
            data_maker=uniform_pmi_dataset_epoch,
            ):
        super().__init__()
        """
        Args:
            📌 X: Dataset to model PMI
            📌 max_interaction_depth: How many interactions we will model
            📌 activation: Activation functions used in network -> string
            📌 embedding_size: This relates to neural network that will model the
            PMI embeddings are generated to simulate missingness during training
            📌 masked_units: This is mask size of each neuron in masked neural
            📌 hidden_units: Hidden units in FFNN
            📌 batch_size: Training batch size
            📌 epochs: How many epochs to train
            📌 max_to_keep: How many checkpoints to save during training w.r.t.
            valid dataset
            📌 num_evaluation_trials: How many times to run the model on valid
            data? Validation simulates missingness so the more you run a better 
            MC estimate
        """
        self.epochs = epochs
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.batch_size = batch_size
        self.num_evaluation_trials = num_evaluation_trials
        self.max_interaction_depth = max_interaction_depth
        self.eps = 1e-7

        input_size = X.shape[-1]
        self.interactions = generate_combinations(
            range(input_size), max_interaction_depth
            )
        self.network = PMINetwork(
            input_size=input_size,
            embedding_size=embedding_size,
            masked_units=masked_units,
            hidden_units=hidden_units,
            activation=activation,
            )

        X = X.astype(np.float32)
        D = X.shape[1]
        X_train, X_val, _, _ = train_test_split(
            X, X, test_size=0.33, random_state=42
            )

        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: data_maker(X_train, batch_size),
            output_signature=((
                tf.TensorSpec(shape=(None, D), dtype=tf.float32),
                tf.TensorSpec(shape=(None, D), dtype=tf.float32)),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )
            ).prefetch(tf.data.AUTOTUNE)
        self.total = sum(1 for _ in self.train_dataset)

        self.valid_dataset = tf.data.Dataset.from_generator(
            lambda: data_maker(X_val, batch_size),
            output_signature=((
                tf.TensorSpec(shape=(None, D), dtype=tf.float32),
                tf.TensorSpec(shape=(None, D), dtype=tf.float32)),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )
            ).prefetch(tf.data.AUTOTUNE)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3, weight_decay=1e-4
            )
        self.filename = datetime.now().strftime("pmi_model_%Y%m%d_%H%M%S")
        self.best_val_loss = tf.Variable(float('inf'), trainable=False)
        self.checkpoint = tf.train.Checkpoint(
            model=self.network,
            optimizer=self.optimizer,
            best_val_loss=self.best_val_loss
            )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.filename, max_to_keep=max_to_keep
            )

    def train(self):
        """
        📌 Learn a neural network to estimate the exp(PMI) values.
        """
        try:
            print('Fitting PMI estimation module.')
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                for step, ((x_batch, m_batch), y_batch) in enumerate(
                        tqdm(
                            self.train_dataset,
                            desc=f"Epoch {epoch+1}",
                            total=self.total
                            )
                        ):
                    with tf.GradientTape() as tape:
                        y_pred = self.network(
                            (x_batch, m_batch), training=True
                            )
                        loss = self.loss_fn(y_batch, y_pred)

                    grads = tape.gradient(
                        loss, self.network.trainable_variables
                        )
                    self.optimizer.apply_gradients(
                        zip(grads, self.network.trainable_variables)
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
        """
        📌 A single validation evaluation. 
        Returns:
           📌 avg_val_loss: performance on validation set
        """
        val_losses = []
        for (x_val, m_val), y_val in self.valid_dataset:
            y_pred_val = self.network(
                (x_val, m_val), training=False
                )
            val_loss = self.loss_fn(y_val, y_pred_val)
            val_losses.append(val_loss.numpy())

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    def load_top_model(self):
        """
        📌 Load the top öodel over multiple validation evaluations.
        """
        saved_checkpoints = self.manager.checkpoints
        checkpoint_losses = []
        for path in tqdm(saved_checkpoints, total=len(saved_checkpoints)):
            self.checkpoint.restore(path).expect_partial()
            print(f"Evaluating {path}")
            total_loss = 0.0
            for i in range(self.num_evaluation_trials):
                loss = self.evaluate_model_on_validation_dataset()
                total_loss += loss
            avg_loss = total_loss / self.num_evaluation_trials
            checkpoint_losses.append((path, avg_loss))

        # Pick the best
        best_ckpt_path, best_loss = min(checkpoint_losses, key=lambda x: x[1])
        print(
            f"\nBest checkpoint: {best_ckpt_path}"
            f"with avg loss: {best_loss:.4f}"
            )
        self.checkpoint.restore(best_ckpt_path).expect_partial()
    
    def inv_exp_pmi(self, x, m):
        """
        📌 Compute inv_exp_pmi for a given m (interaction) value.
        📌 Args:
            x: batch of instances that you want to compute it w.r.t.
            m: masking matrix (if a value is not present in interaction)
            its m-value is 0
            
        📌 Returns:
            inv_exp_pmi_vals: inverse exponential pmi values
        """
        inv_exp_pmi_vals \
            = self.network(
                (
                    tf.convert_to_tensor(x),
                    tf.convert_to_tensor(m)
                    ),
                training=False
                )

        inv_exp_pmi_vals = inv_exp_pmi_vals / (
            self.eps + 1 - inv_exp_pmi_vals
            )
        inv_exp_pmi_vals = 1 / (inv_exp_pmi_vals + 1e-2)
        return tf.stop_gradient(inv_exp_pmi_vals)

    def inv_exp_pmi_dict(self, x, test=False):
        """
        For JOAK, use this.
        📌 Args:
            📌 x: typically the values you will feed into your OAK-Kernel

        📌 Returns:
            📌 inv_exp_pmi_dict: 
                inverse exponential pmi values given an input instance x.
                keys are interaction values (tuple) and values are tf arrays.
            📌 i.e., N by 2^D PMI values (per-instance by per-interaction)
        """
        inv_exp_pmi_dict = dict()
        for interaction in [[]] + self.interactions:
            m = np.zeros(x.shape)
            m[:,interaction] = 1
            #returns logits: log (p/1-p)
            if len(interaction) <= 1:
                if test:
                    inv_exp_pmi_dict[tuple(interaction)] = self.inv_exp_pmi(
                        x, m
                        )
                else:
                    inv_exp_pmi_dict[tuple(interaction)] \
                        = tf.convert_to_tensor(np.ones((x.shape[0], 1)))
            else:
                inv_exp_pmi_dict[tuple(interaction)] = self.inv_exp_pmi(
                        x, m
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
    model = NeuralPMIModel(
        X=X,
        data_maker=uniform_pmi_dataset_epoch
        )
    model.train()

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x1 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack([grid_x0.ravel(), grid_x1.ravel()], axis=1)
    pmi_points = model.inv_exp_pmi_dict(grid_points, test=True) #for testing
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
