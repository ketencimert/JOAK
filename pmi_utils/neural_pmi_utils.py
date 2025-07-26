# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 21:27:35 2025

@author: Mert
"""

import numpy as np
import itertools
import tensorflow as tf
from pmi_utils.shared_pmi_utils import topk_numpy

def masked_fill(tensor, mask, value):
    return tf.where(mask, tf.fill(tf.shape(tensor), value), tensor)

def create_resulting_mask(input_size, output_size, combinations):
    #Define the last mask
    comb_size = len(combinations)
    comb_range = range(comb_size)
    mask = np.zeros((input_size, output_size))
    choice = np.random.choice(
        comb_range,
        output_size-comb_size,
        replace=True
        )
    choices = np.sort(np.concatenate([comb_range, choice]))
    for i, choice in enumerate(choices):
        for c in combinations[choice]:
            mask[c,i] = 1
    return mask

def generate_masks(input_size, embedding_size, hidden):

    combination_order = 1

    features = list(range(input_size))
    combinations = list(
        itertools.combinations(features, combination_order)
        )

    #this is a must in general
    assert all([h>=len(combinations) for h in hidden])

    layers = [input_size] + hidden
    first_mask = create_resulting_mask(
        input_size=layers[0],
        output_size=layers[1],
        combinations=combinations,
        )

    masks = [first_mask]
    layers = list(layers)
    resulting_mask = None
    for h0, h1 in zip(layers[:-1][1:], layers[1:][1:]):
        # print(h0, h1)
        resulting_mask = create_resulting_mask(
            input_size,
            h1,
            combinations
            ) #want to be in the form of

        second_last_mask = []
        for i in range(h1):
            ones = []
            zeros = []
            topk = topk_numpy(
                1*(resulting_mask[:,i] !=0), resulting_mask.shape[0]
                )
            for value, index in zip(topk[0], topk[1]):
                if value != 0:
                    ones.append(index.item())
                else:
                    zeros.append(index.item())
            candidate = first_mask[ones].sum(0) != 0
            fltr = first_mask[zeros].sum(0) == 0
            second_last_mask.append((fltr*candidate*1.).reshape(-1,1))

        second_last_mask = np.concatenate(second_last_mask, 1)
        first_mask = np.matmul(first_mask, second_last_mask)
        masks.append(second_last_mask)
    second_last_mask = []
    if resulting_mask is None:
        resulting_mask = masks[0]
    for row in resulting_mask:
        row = row.reshape(-1,1).repeat(embedding_size, -1)
        second_last_mask.append(row)
    second_last_mask = np.concatenate(second_last_mask, -1)
    masks.append(second_last_mask)
    return masks

class MaskedDense(tf.keras.layers.Layer):
    def __init__(self, units, mask, activation=None, use_bias=True):
        super().__init__()
        self.units = units
        self.mask = tf.constant(mask, dtype=tf.float32)  # fixed binary mask
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="weights"
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias"
            )
        else:
            self.b = None

    def call(self, inputs):
        # masked_w = self.w * self.mask  # element-wise masking
        masked_w = masked_fill(self.w, self.mask==0, 0.)
        output = tf.matmul(inputs, masked_w)
        if self.use_bias:
            output = output + self.b
        if self.activation:
            output = self.activation(output)
        return output

class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_units, activation):
        super(SimpleAttention, self).__init__()
        self.input_size = input_size
        # Projection to compute attention scores
        self.attn_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    h, 
                    kernel_initializer='glorot_uniform', 
                    activation=activation
                    ) for h in hidden_units
                ] + [tf.keras.layers.Dense(1)]
            )

    def call(self, x):
        """
        x: Tensor of shape (N, D, E)
        Returns: Tensor of shape (N, E)
        """
        # Compute attention logits (N, D, 1)
        attn_logits = self.attn_proj(x) / (self.input_size**0.5) # shape: (N, D, 1)
        attn_weights = tf.nn.softmax(attn_logits, axis=1)  # softmax over D
        # Weighted sum over D: shape (N, E)
        weighted_sum = tf.reduce_sum(attn_weights * x, axis=1)
        return weighted_sum

class PMINetwork(tf.keras.Model):
    def __init__(
            self, 
            input_size=2, 
            embedding_size=20,
            masked_units=[64, 64], 
            hidden_units=[64, 64],
            activation='elu'
            ):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        self.encoder = tf.keras.Sequential(
            [
                MaskedDense(
                    m.shape[1], mask=m, activation=activation
                    ) for m in generate_masks(
                        input_size,
                        embedding_size,
                        masked_units
                        )
            ]
            )
        self.placeholder = self.add_weight(
            shape=(1, input_size, embedding_size),
            initializer='glorot_uniform', 
            trainable=True,
            name='placeholder'
            )

        self.aggregator = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    h, 
                    kernel_initializer='glorot_uniform', 
                    activation=activation
                    ) for h in hidden_units
                ] + [tf.keras.layers.Dense(1)]
            )
        self.attention = SimpleAttention(
            self.input_size, hidden_units, activation
            )
        
    def call(self, inputs):
        x, m = inputs

        # Step 1: Apply shared encoder to each feature independently
        x_encoded = self.encoder(x)  # shape: (B, D, embed_dim)
        x_encoded = tf.reshape(
            x_encoded, [-1, self.input_size, self.embedding_size]
            )

        # Step 2: Aggregate (sum over features)
        m_expand = tf.repeat(
            tf.expand_dims(m, axis=-1), repeats=x_encoded.shape[-1], axis=-1
            )
        B = tf.shape(x_encoded)[0]
        placeholder_tiled = tf.tile(self.placeholder, [B, 1, 1])
        # shape: (B, D, E)

        x_encoded = x_encoded * m_expand + (1 - m_expand) * placeholder_tiled
        x_agg = self.attention(x_encoded)  # shape: (B, embed_dim)
        p = self.aggregator(x_agg)
        p = 0.5 * (p / (1 + tf.abs(p)) + 1)
        return p