# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import gpflow
import numpy as np
import tensorflow as tf
from functools import reduce
from typing import List, Optional, Tuple, Type
from joak.input_measures import (
    EmpiricalMeasure,
    GaussianMeasure,
    MOGMeasure,
)
from joak.ortho_binary_kernel import OrthogonalBinary
from joak.ortho_categorical_kernel import OrthogonalCategorical
from joak.ortho_rbf_kernel import OrthogonalRBFKernel
from tensorflow_probability import bijectors as tfb

from pmi_model import PMIModel
# -

def bounded_param(low: float, high: float, param: float) -> gpflow.Parameter:
    """
    :param low: lower bound of the parameter
    :param high: upper bound of the parameter
    :param param: initial parameter value
    :return: tfp Parameter with optimization bounds
    """
    sigmoid = tfb.Sigmoid(low=tf.cast(low, tf.float64), high=tf.cast(high, tf.float64))
    parameter = gpflow.Parameter(param, transform=sigmoid, dtype=tf.float64)
    return parameter


class JOAKKernel(gpflow.kernels.Kernel):
    """
    Compute OAK kernel
    :param base_kernels: List of base kernel classes for constructing durrande kernel for non-binary inputs. To be deleted
        after initialisation
    :param num_dims: dimensionality of input data
    :param max_interaction_depth: maximum order of interactions
    :param active_dims: pass active_dims in case there is not one kernel per input dim
                (e.g. in the distributional case)
    :param constrain_orthogonal: whether to use orthogonal kernel or not
    :param p0: list of probability measure for binary kernels, for continuous/categorical kernel, it is set to None
        :param p: list of probability measure for categorical kernels, for continuous/binary kernel, it is set to None
    :param lengthscale_bounds: list of lower and upper bounds for lengthscale, this is to upper bound the normalisation in high dimensions
                Code currently supports a common bound for all lengthscales. In the future we could extend it to
                have different bounds per dimension but this is probably not needed due our use of input normalisation.
    :param empirical_locations: list of locations for empirical measures, if using Gaussian measure, this is set to None
    :param empirical_weights: list of weights for empirical measures, if using Gaussian measure, this is None
    :param gmm_measures: Gaussian mixture model measure for continuous inputs.
    :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses original constrained kernel \prod_i(1+k_i); if True, this is the OAK kernel
    :return: OAK kernel
    """

    def __init__(
        self,
        base_kernels: List[Type[gpflow.kernels.Kernel]],
        pmi_model: PMIModel,
        num_dims: int,
        max_interaction_depth: int,
        active_dims: Optional[List[List[int]]] = None,
        constrain_orthogonal: bool = False,
        p0: Optional[List[float]] = None,
        p: Optional[List[float]] = None,
        lengthscale_bounds: Optional[List[float]] = None,
        empirical_locations: Optional[List[float]] = None,
        empirical_weights: Optional[List[float]] = None,
        gmm_measures: Optional[List[MOGMeasure]] = None,
        share_var_across_orders: Optional[bool] = True,
    ):
        super().__init__(active_dims=range(num_dims))
        if active_dims is None:
            active_dims = [[dim] for dim in range(num_dims)]
        # assert that active dims doesn't contain duplicates and doesn't exceed the total num_dims
        flat_dims = [dim for sublist in active_dims for dim in sublist]
        assert max(flat_dims) <= num_dims, "Active dims exceeding num dims."
        assert len(flat_dims) == len(
            np.unique(flat_dims)
        ), "Active dims contains duplicates."
        delta2 = 1  # prior measure process variance hardcoded to 1
        # set up kernels (without parameters for variance)
        self.base_kernels, self.max_interaction_depth = (
            base_kernels,
            max_interaction_depth,
        )
        self.share_var_across_orders = share_var_across_orders
        
        self.pmi_model = pmi_model        
        
        # p0 is a list of probability measures for binary kernels, set to None if it is not binary
        if p0 is None:
            p0 = [None] * len(active_dims)

        # p is a list of probability measures for categorical kernels, set to None if it is not categorical
        if p is None:
            p = [None] * len(active_dims)

        if constrain_orthogonal:
            if empirical_locations is None:
                assert (
                    empirical_weights is None
                ), "Cannot have weights without locations"
                empirical_locations = [None] * len(active_dims)
                empirical_weights = [None] * len(active_dims)
            else:
                if empirical_weights is not None:
                    location_shapes = [
                        len(empirical_locations[dim])
                        if empirical_locations[dim] is not None
                        else None
                        for dim in range(len(active_dims))
                    ]
                    location_weights = [
                        len(empirical_weights[dim])
                        if empirical_weights[dim] is not None
                        else None
                        for dim in range(len(active_dims))
                    ]
                    print(location_shapes)
                    assert (
                        location_shapes == location_weights
                    ), f"Shape of empirical measure locations {location_shapes} do not match weights {location_weights}"

            if gmm_measures is None:
                gmm_measures = [None] * len(active_dims)

            self.kernels = []

            for dim in range(len(active_dims)):
                # len(active_dims) can be < num_dims if some inputs are grouped
                if (
                    empirical_locations[dim] is not None
                    and gmm_measures[dim] is not None
                ):
                    raise ValueError(
                        f"Both empirical and GMM measure defined for input {dim}"
                    )

                if (p0[dim] is None) and (p[dim] is None):
                    if empirical_locations[dim] is not None:
                        k = OrthogonalRBFKernel(
                            base_kernels[dim](),
                            EmpiricalMeasure(
                                empirical_locations[dim], empirical_weights[dim]
                            ),
                            active_dims=active_dims[dim],
                        )
                    elif gmm_measures[dim] is not None:
                        k = OrthogonalRBFKernel(
                            base_kernels[dim](),
                            measure=gmm_measures[dim],
                            active_dims=active_dims[dim],
                        )

                    else:
                        # Continuous input with Gaussian measure
                        k = OrthogonalRBFKernel(
                            base_kernels[dim](),
                            GaussianMeasure(0, delta2),
                            active_dims=active_dims[dim],
                        )
                        if share_var_across_orders:
                            k.base_kernel.variance = tf.ones(
                                1, dtype=gpflow.config.default_float()
                            )

                    if lengthscale_bounds is not None:
                        k.base_kernel.lengthscales = bounded_param(
                            lengthscale_bounds[0], lengthscale_bounds[1], 1
                        )
                elif p[dim] is not None:
                    assert base_kernels[dim] is None
                    k = OrthogonalCategorical(
                        p=p[dim],
                        active_dims=active_dims[dim],
                    )
                    if share_var_across_orders:
                        k.variance = tf.ones(1, dtype=gpflow.config.default_float())
                else:
                    assert base_kernels[dim] is None
                    k = OrthogonalBinary(
                        p0=p0[dim],
                        active_dims=active_dims[dim],
                    )
                    if share_var_across_orders:
                        k.variance = tf.ones(1, dtype=gpflow.config.default_float())

                self.kernels.append(k)
        # unconstrained kernel with the additive model structure
        else:
            assert (
                empirical_locations is None
            ), "Cannot have empirical locations without orthogonal constraint"
            assert (
                empirical_weights is None
            ), "Cannot have empirical weights without orthogonal constraint"

            self.kernels = []
            for dim in range(len(active_dims)):
                # point cases
                if p0[dim] is None:
                    k = base_kernels[dim](active_dims=active_dims[dim])

                else:
                    assert base_kernels[dim] is None
                    k = OrthogonalBinary(p0=p0[dim], active_dims=active_dims[dim])
                if share_var_across_orders:
                    k.variance = tf.ones(1, dtype=gpflow.config.default_float())
                self.kernels.append(k)
        # add parameters to control the variances for various interaction orders (+1 for bias/constant term)
        if self.share_var_across_orders:
            self.variances = [
                gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
                for _ in range(max_interaction_depth + 1)
            ]
        else:
            # only have additional variance for the constant kernel
            self.variances = [
                gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
            ]

    def compute_kernel_weights(self, X, X2=None, replicate=True):
        if X2 is None:
            if replicate:
                inputs = tf.concat([X, X], 0)
            else:
                inputs = X
        else:
            inputs = tf.concat([X, X2], 0)
        inv_exp_pmi_dict = self.pmi_model.compute_kernel_weights(inputs)
        return inv_exp_pmi_dict

    def compute_additive_terms(self, kernel_matrices, inv_exp_pmi_dict):
        """
        Given a list of tensors (kernel matrices), compute a new list
        containing all products up to order self.max_interaction_depth.

        Example:
          input: [a, b, c, d]
          output: [1, (a+b+c+d), (ab+ac+ad+bc+bd+cd), (abc+abd+acd+bcd), abcd)]
        Does not use the Girard Newton identity, as found in Duvenaud et al "Additive GPs".
        """
        D = len(kernel_matrices)
        if len(kernel_matrices[0].shape) > 1:
            N, M = kernel_matrices[0].shape
            diag = False
        else:
            N = kernel_matrices[0].shape[0]
            diag = True

        max_interaction_depth = self.max_interaction_depth
        result_by_order = [
            tf.zeros_like(
                kernel_matrices[0]
                ) for _ in range(max_interaction_depth + 1)
            ]
        for r in range(1, max_interaction_depth + 1):
            for subset in itertools.combinations(range(D), r):
                key = tuple(sorted(subset))
                #this needs to return N*M by 1
                inv_exp_pmi = inv_exp_pmi_dict[key]
                if not diag:
                    inv_exp_pmi1, inv_exp_pmi2 = inv_exp_pmi[:N],\
                    inv_exp_pmi[N:]
                    inv_exp_pmi = inv_exp_pmi1 @ tf.transpose(inv_exp_pmi2)
                else:
                    inv_exp_pmi = tf.squeeze(inv_exp_pmi**2, -1)
                #if key in interaction_weights:
                # weight = interaction_weights[key]
                term = tf.ones_like(kernel_matrices[0])
                for i in subset:
                    term *= kernel_matrices[i]
                result_by_order[r] += term * tf.cast(inv_exp_pmi, tf.float64)
                    # result_by_order[r] += weight * term
        # constant term (optional)
        result_by_order[0] = tf.ones_like(kernel_matrices[0])
        return result_by_order
        """
        Given a list of tensors (kernel matrices), compute a new list
        containing all products up to order self.max_interaction_depth.

        Example:
          input: [a, b, c, d]
          output: [1, (a+b+c+d), (ab+ac+ad+bc+bd+cd), (abc+abd+acd+bcd), abcd)]
        Does not use the Girard Newton identity, as found in Duvenaud et al "Additive GPs".
        """
        D = len(kernel_matrices)
        N = kernel_matrices[0].shape
        max_interaction_depth = self.max_interaction_depth
        result_by_order = [
            tf.zeros_like(
                kernel_matrices[0]
                ) for _ in range(max_interaction_depth + 1)
            ]
        for r in range(1, max_interaction_depth + 1):
            for subset in itertools.combinations(range(D), r):
                key = tuple(sorted(subset))
                #this needs to return N*M by 1
                inv_exp_pmi = inv_exp_pmi_dict[key]

                inv_exp_pmi1, inv_exp_pmi2 = inv_exp_pmi[:N],\
                inv_exp_pmi[N:]
                inv_exp_pmi = tf.squeeze(inv_exp_pmi1 * inv_exp_pmi2, -1)
                #if key in interaction_weights:
                # weight = interaction_weights[key]
                term = tf.ones_like(kernel_matrices[0])
                for i in subset:
                    term *= kernel_matrices[i]
                result_by_order[r] += term * tf.cast(inv_exp_pmi, tf.float64)
                    # result_by_order[r] += weight * term
        # constant term (optional)
        result_by_order[0] = tf.ones_like(kernel_matrices[0])
        return result_by_order

    def K(self, X, X2=None):
        kernel_matrices = [
            k(X, X2) for k in self.kernels
        ]  # note that active dims gets applied by each kernel

        inv_exp_pmi_dict = self.compute_kernel_weights(X, X2)
        additive_terms = self.compute_additive_terms(
            kernel_matrices, inv_exp_pmi_dict
            )
        if self.share_var_across_orders:
            return reduce(
                tf.add,
                [sigma2 * k for sigma2, k in zip(self.variances, additive_terms)],
            )
        else:
            # add constant kernel
            return reduce(
                tf.add, [self.variances[0] * additive_terms[0]] + additive_terms[1:]
            )

    def K_diag(self, X):
        kernel_diags = [k.K_diag(k.slice(X)[0]) for k in self.kernels]

        inv_exp_pmi_dict = self.compute_kernel_weights(X, replicate=False)
        additive_terms = self.compute_additive_terms(
            kernel_diags, inv_exp_pmi_dict
            )
        if self.share_var_across_orders:
            return reduce(
                tf.add,
                [sigma2 * k for sigma2, k in zip(self.variances, additive_terms)],
            )
        else:
            return reduce(
                tf.add, [self.variances[0] * additive_terms[0]] + additive_terms[1:]
            )


class KernelComponenent(gpflow.kernels.Kernel):
    def __init__(
        self,
        joak_kernel: JOAKKernel,
        iComponent_list: List[int],
        share_var_across_orders: Optional[bool] = True,
    ):
        # Orthogonal kernel + interactions kernel
        # sort out active_dims - it must be a list of integers
        super().__init__(active_dims=joak_kernel.active_dims)
        self.joak_kernel = joak_kernel
        self.iComponent_list = iComponent_list
        self.share_var_across_orders = share_var_across_orders
        self.kernels = [
            k
            for i, k in enumerate(self.joak_kernel.kernels)
            if i in self.iComponent_list
        ]

    def K(self, X, X2=None):
        inv_exp_pmi_dict = self.joak_kernel.compute_kernel_weights(X, X2)
        N = X.shape[0]
        inv_exp_pmi = inv_exp_pmi_dict[self.iComponent_list]
        inv_exp_pmi1, inv_exp_pmi2 = inv_exp_pmi[:N],\
                    inv_exp_pmi[N:]
        inv_exp_pmi = inv_exp_pmi1 @ tf.transpose(inv_exp_pmi2) #W
        inv_exp_pmi = inv_exp_pmi / tf.expand_dim(tf.diag(inv_exp_pmi), -1)
        if len(self.iComponent_list) == 0:
            shape = (
                [tf.shape(X)[0], tf.shape(X)[0]]
                if X2 is None
                else [tf.shape(X)[0], tf.shape(X2)[0]]
            )
            return self.joak_kernel.variances[0] * tf.ones(
                shape, dtype=gpflow.default_float()
            )  # start with constant term
        else:
            # element wise product
            # compute kernel in iComponent_list only
            n_order = len(self.iComponent_list)  # [0, 1]
            k_mats = [k(X, X2) for k in self.kernels]
            variances_n = (
                self.joak_kernel.variances[n_order]
                if self.share_var_across_orders
                else 1
            )
            return variances_n * tf.reduce_prod(k_mats, axis=0) \
                * tf.cast(inv_exp_pmi, tf.float64)

    def K_diag(self, X):
        inv_exp_pmi_dict = self.joak_kernel.compute_kernel_weights(
            X, replicate=False
            )
        inv_exp_pmi = inv_exp_pmi_dict[self.iComponent_list]
        inv_exp_pmi = tf.squeeze(inv_exp_pmi**2, -1)
        if len(self.iComponent_list) == 0:
            return self.joak_kernel.variances[0] * tf.ones(
                tf.shape(X)[0], dtype=gpflow.default_float()
            )  # start with constant term
        else:
            n_order = len(self.iComponent_list)
            k_mats = [k.K_diag(k.slice(X)[0]) for k in self.kernels]
            variances_n = (
                self.joak_kernel.variances[n_order]
                if self.share_var_across_orders
                else 1
            )
            return variances_n * tf.reduce_prod(k_mats, axis=0) \
                * tf.cast(inv_exp_pmi, tf.float64)


def get_list_representation(
    kernel: JOAKKernel,
    num_dims: int,
    share_var_across_orders: Optional[bool] = True,
) -> Tuple[List[List[int]], List[KernelComponenent]]:
    """
    Construct kernel list representation of OAK kernel
    """
    assert isinstance(kernel, JOAKKernel)
    selected_dims = []
    kernel_list = []
    selected_dims.append([])  # no dimensions for constant term
    kernel_list.append(
        KernelComponenent(kernel, [], share_var_across_orders=share_var_across_orders)
    )  # add constant
    if kernel.max_interaction_depth > 0:
        for ii in range(kernel.max_interaction_depth + 1):
            if ii > 0:
                tmp = [
                    list(tup) for tup in itertools.combinations(np.arange(num_dims), ii)
                ]
                selected_dims = selected_dims + tmp

                for jj in range(len(tmp)):
                    kernel_list.append(KernelComponenent(kernel, tmp[jj]))

    return selected_dims, kernel_list
