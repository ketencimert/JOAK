# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:01:21 2025

@author: Mert
"""

import numpy as np
from itertools import combinations

def topk_numpy(arr, k, axis=-1, largest=True, sorted=True):
    """
    Args:
        📌 Numpy implementation of torch.topk
        📌 arr: input array
        📌 topk: number of 'top' elements to return
        📌 axis: axis which the 'top' elements are returned

    Returns:
        📌 topk_values_unsorted, topk_indices_unsorted same as torch
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

def generate_combinations(lst, C):
    """
    Args:
        📌 Generate all possible combinations of the elements in a list.
        📌 lst: List of indices. e.g. [0,1,2]
        📌 C: Maximum order of combinations

    Returns:
        list of list of interactions.
        📌 e.g., if input is [0,1,2] output will be
        [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
    """
    all_combos = []
    for r in range(1, C + 1):
        all_combos.extend(combinations(lst, r))
    return [list(c) for c in all_combos]

def simple_dataset_epoch(X, batch_size=256):
    """
    Args:
        📌 Just for training loop.
        📌 X: Dataset to batch
        📌 batch_size : Batch size used during training

    Returns:
        📌 A batch of X samples
    """
    N, D = X.shape
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        yield X_batch.astype(np.float32)


def uniform_pmi_dataset_epoch(X, batch_size=256):
    """
    Args:
        📌  This function assumes a uniform distribution over feature masking.
        That is, for [0,1,2] we have [0,0,0] -> 1/8, [0,0,1] -> 1/8, [0,1,0] ->1/8
        [0,1,1] -> 1/8 and so forth.
        📌 X: Dataset to batch
        📌 batch_size : Batch size used during training
    Returns:
        📌 A batch of samples to train the neural model.
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
    Args:
        📌  This function assumes a uniform distribution over
        feature masking sizes.
        That is, uniform distribution over "coalition" sizes. For [0,1,2], 
        we have
        size(0) -> 1/4, size(1) -> 1/4, size(2) -> 1/4, size(3) -> 1/4.
        📌 X: Dataset to batch
        📌 batch_size : Batch size used during training
    Returns:
        📌 A batch of samples to train the neural model.
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
