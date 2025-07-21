from datetime import datetime
from itertools import combinations

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from pmi_network import PMINetwork

def sigmoid(z):
    """
    📌 Computes the sigmoid
    """
    return 1/(1 + np.exp(-z))

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

def topk_numpy(arr, k, axis=-1, largest=True, sorted=True):
    """
    📌 Numpy implementation of torch.topk
    
    📌 arr: input array
    
    📌 topk: number of 'top' elements to return
    
    📌 axis: axis which the 'top' elements are returned
    """
    if largest:
        partitioned_indices = np.argpartition(-arr, k-1, axis=axis)
    else:
        partitioned_indices = np.argpartition(arr, k-1, axis=axis)

    topk_indices_unsorted = np.take(
        partitioned_indices, np.arange(k), axis=axis
        )

    topk_values_unsorted = np.take_along_axis(
        arr, topk_indices_unsorted, axis=axis
        )

    if sorted:
        sort_order \
            = np.argsort(
                -topk_values_unsorted if largest else topk_values_unsorted,
                axis=axis
                )
        topk_values \
            = np.take_along_axis(
                topk_values_unsorted, sort_order, axis=axis
                )
        topk_indices \
            = np.take_along_axis(
                topk_indices_unsorted, sort_order, axis=axis
                )
        return topk_values, topk_indices
    else:
        return topk_values_unsorted, topk_indices_unsorted

def uniform_pmi_dataset_epoch(X, batch_size=256):
    """
    📌  This function assumes a uniform distribution over feature masking.
    That is, for [0,1,2] we have [0,0,0] -> 1/8, [0,0,1] -> 1/8, [0,1,0] ->1/8
    [0,1,1] -> 1/8 and so forth.
    
    📌 X: Dataset to batch
    
    📌 batch_size : Batch size used during training
    """
    N, D = X.shape
    # Shuffle dataset indices once per epoch
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        x_joint = X[batch_idx]
        # Create marginal sample by shuffling all but one randomly chosen
        # feature
        x_marg = x_joint.copy()
        # Shuffle each column independently
        perms = np.argsort(np.random.rand(x_joint.shape[0], D), axis=0)
        # Use advanced indexing to apply permutations per column
        x_marg = x_marg[perms, np.arange(D)]
        # Stack joint and marginal, label joint=1 and marginal=0
        X_batch = np.concatenate([x_joint, x_marg], axis=0)
        y_batch = np.concatenate(
            [np.ones(len(x_joint)), np.zeros(len(x_marg))], axis=0
            )
        m_batch = np.random.binomial(1, 1/2, x_joint.shape)
        m_batch = np.concatenate([m_batch, m_batch], 0)
        yield (
            X_batch.astype(np.float32), m_batch.astype(np.float32)
            ), y_batch.astype(np.float32)

def shapley_pmi_dataset_epoch(X, batch_size=256):
    """
    📌  This function assumes a uniform distribution over feature masking sizes.
    That is, uniform distribution over "coalition" sizes. For [0,1,2], we have
    size(0) -> 1/4, size(1) -> 1/4, size(2) -> 1/4, size(3) -> 1/4.

    📌 X: Dataset to batch

    📌 batch_size : Batch size used during training
    """
    N, D = X.shape
    # Shuffle dataset indices once per epoch
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        x_joint = X[batch_idx]
        # Create marginal sample by shuffling all but one randomly chosen
        # feature
        x_marg = x_joint.copy()
        # Shuffle each column independently
        perms = np.argsort(np.random.rand(x_joint.shape[0], D), axis=0)
        # Use advanced indexing to apply permutations per column
        x_marg = x_marg[perms, np.arange(D)]
        # Stack joint and marginal, label joint=1 and marginal=0
        X_batch = np.concatenate([x_joint, x_marg], axis=0)
        y_batch = np.concatenate(
            [np.ones(len(x_joint)), np.zeros(len(x_marg))], axis=0
            )

        feature_idx_init = np.zeros_like(x_joint)[:,0]
        feature_idx = np.expand_dims(feature_idx_init, -1)

        permutation = np.argsort(
            np.random.normal(size=(
                x_joint.shape[0],
                x_joint.shape[-1] + 1)),
            -1
        )

        arange = np.repeat(
            np.expand_dims(np.arange(permutation.shape[-1]), 0),
            permutation.shape[0], 0
            )
        pointer = arange <= np.argmax(
            (permutation == feature_idx) * 1., -1
        ).reshape(-1, 1)
        p_sorted = topk_numpy(
            -permutation, permutation.shape[-1], -1, sorted=True
            )[1]
        m_batch = np.concatenate(
            [
                np.diag(
                    pointer[:, p_sorted[:, i]]
                ).reshape(-1, 1) for i in range(
                p_sorted.shape[-1]
            )
            ], -1
                    )[:,1:]
        m_batch = np.concatenate([m_batch, m_batch], 0)

        yield (
            X_batch.astype(np.float32), m_batch.astype(np.float32)
            ), y_batch.astype(np.float32)

class PMIModel(tf.keras.Model):
    def __init__(
            self,
            X: np.asarray,
            max_interaction_depth=2,
            activation='elu',
            embedding_size=20,
            masked_units=[200, 200],
            hidden_units=[64, 64, 64],
            batch_size=1024,
            epochs=1000,
            max_to_keep=20,
            num_evaluation_trials=25, #reduce the variance 5 times
            data_maker=uniform_pmi_dataset_epoch,
            ):
        super().__init__()
        """
        📌 X: Dataset to model PMI -> N by D numpy array

        📌 max_interaction_depth: How many interactions we will model 
        -> integer
        
        📌 activation: Activation functions used in network -> string
        
        📌 embedding_size: This relates to neural network that will model the 
        PMI embeddings are generated to simulate missingness during training
        -> integer

        📌 masked_units: This is mask size of each neuron in masked neural 
        network -> list of integers

        📌 hidden_units: Hidden units in FFNN
        -> list of integers

        📌 batch_size: Training batch size -> integer

        📌 epochs: How many epochs to train -> integer

        📌 max_to_keep: How many checkpoints to save during training w.r.t. 
        valid dataset

        📌 num_evaluation_trials: How many times to run the model on valid 
        data? Validation simulates missingness so the more you run a better MC 
        estimate -> integer
        """
        self.epochs = epochs
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.batch_size = batch_size
        self.num_evaluation_trials = num_evaluation_trials
        self.max_interaction_depth = max_interaction_depth

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

        self.optimizer = tf.keras.optimizers.Adam()
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
        # Get all saved checkpoints
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

    def exp_pmi_values_(self, x):
        """
        Run this for testing.
        
        📌 input: typically the values you will feed into your OAK-Kernel

        📌 output: exponential pmi values given an input instance x.
        if x is 3 dimensional and your are modeling all interactions than
        columns will represent:
        [null, 0, 1, 2, (0,1), (0,2), (1,2), (1,2,3)]
        
        📌 i.e., N by 2^D PMI values (per-instance by per-interaction)
        """
        exp_pmi = []
        for interaction in [[]] + self.interactions:
            m = np.zeros_like(x)
            m[:,interaction] = 1
            #returns logits: log (p/1-p)
            exp_pmi_vals = np.exp(
                self.network.predict(
                    (x, tf.convert_to_tensor(m)),
                    batch_size=self.batch_size,
                    )
                )
            exp_pmi.append(exp_pmi_vals)
        return np.concatenate(exp_pmi, -1)

    def exp_pmi_values(self, x):
        """
        For OAK, use this.
        
        📌 input: typically the values you will feed into your OAK-Kernel

        📌 output: exponential pmi values given an input instance x.
        if x is 3 dimensional and your are modeling all interactions than
        columns will represent:
        [null, 0, 1, 2, (0,1), (0,2), (1,2), (1,2,3)]
        
        📌 i.e., N by 2^D PMI values (per-instance by per-interaction)
        """
        exp_pmi = []
        for interaction in [[]] + self.interactions:
            m = np.zeros_like(x)
            m[:,interaction] = 1
            #returns logits: log (p/1-p)
            if len(interaction) <= 1:
                exp_pmi_vals = np.ones((x.shape[0], 1))
            else:
                exp_pmi_vals = np.exp(
                    self.network.predict(
                        (x, tf.convert_to_tensor(m)),
                        batch_size=self.batch_size,
                        )
                    )
            exp_pmi.append(exp_pmi_vals)
        return np.concatenate(exp_pmi, -1)


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
        data_maker=shapley_pmi_dataset_epoch
        )
    model.train()

    # Evaluate PMI on a grid
    x0 = np.linspace(X[:,0].min() * 1.4, X[:,0].max() * 1.4, 200)
    x1 = np.linspace(X[:,1].min() * 1.4, X[:,1].max() * 1.4, 200)
    grid_x0, grid_x1 = np.meshgrid(x0, x1)
    grid_points = np.stack([grid_x0.ravel(), grid_x1.ravel()], axis=1)
    
    pmi_points = model.exp_pmi_values_(grid_points) #for testing
    for points in pmi_points.T:
        # Predict probabilities and compute PMI
        exp_pmi_vals = points
        pmi_grid = exp_pmi_vals.reshape(grid_x0.shape)
        # Plot the estimated PMI
        plt.figure(figsize=(6, 5))
        vmin = pmi_grid.min() - 1e-6
        vmax = pmi_grid.max() + 1e-6
        vcenter = 0.5 * (vmin + vmax)
        divnorm = mcolors.TwoSlopeNorm(
            vmin=vmin,
            vcenter=vcenter, 
            vmax=vmax
            )
        plt.contourf(
            grid_x0, grid_x1, pmi_grid, norm=divnorm, levels=50, cmap='viridis'
            )
        plt.colorbar(label='PMI Estimate')
        plt.title(
            'Estimated PMI(x₀; x₁) via Discriminator (Shuffled In-Model)'
            )
        plt.xlabel('x₀')
        plt.ylabel('x₁')
        plt.tight_layout()
        plt.show()
