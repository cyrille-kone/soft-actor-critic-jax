import os
import jax
import chex
import pickle
import haiku as hk
import jax.numpy as jnp
from typing import Sequence

from utils import logger

"""
Critic, Value and Actor networks
"""

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class CustomMLP(hk.nets.MLP):
    """
    General MLP class for checkpoint saving/loading.
    Value, Critic and Actor networks inherit this class.

    takes an input of size self._expected_input_dims
    """

    def __init__(self, output_sizes: Sequence[int],
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None) -> None:

        if non_linearity == 'relu':
            activation = jax.nn.relu
        else:
            raise NotImplemented

        super().__init__(output_sizes=output_sizes, activation=activation)
        self._expected_input_dims = None  # should be set in child classes

    def __call__(self, x: chex.Array) -> chex.Array:
        assert x.shape[-1] == self._expected_input_dims,\
        f"input dimension {x.shape} doesn't match expected dimension [batch_size, {self._expected_input_dims}]"
        return super().__call__(x)


class CriticNetwork(CustomMLP):
    def __init__(self, obs_dims: int, action_dims: int,
                 hidden_output_dims=(256, 256),
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None):
        super().__init__(
            output_sizes=(*hidden_output_dims, 1),
            non_linearity=non_linearity,
            chkpt_dir=chkpt_dir
        )

        self._expected_input_dims = obs_dims+action_dims  # to assert correct input dims


class ValueNetwork(CustomMLP):
    def __init__(self, obs_dims: int,
                 hidden_output_dims=(256, 256),
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None):
        super().__init__(
            output_sizes=(*hidden_output_dims, 1),
            non_linearity=non_linearity,
            chkpt_dir=chkpt_dir
        )

        self._expected_input_dims = obs_dims  # to assert correct input dims


class ActorNetwork(CustomMLP):
    def __init__(self, obs_dims: int, action_dims: int,
                 hidden_output_dims=(256, 256),
                 non_linearity='relu',
                 chkpt_dir=None):

        super().__init__(
            output_sizes=(*hidden_output_dims, 2*action_dims),
            non_linearity=non_linearity,
            chkpt_dir=chkpt_dir
        )

        self._expected_input_dims = obs_dims  # to assert correct input dims

    def __call__(self, state: chex.Array) -> chex.Array:
        h = super().__call__(state)
        mu, log_sigma = jnp.split(h, 2, axis=-1)

        log_sigma = jnp.clip(log_sigma, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)  # to prefent -inf values
        return mu, log_sigma

