import os
import jax
import chex
import pickle
import logging
import haiku as hk
import jax.numpy as jnp
from typing import Sequence

"""
Critic, Value and Actor networks
"""


# Inherits from MLP so no need to reimplement forward/__call__
class CustomMLP(hk.nets.MLP):
    """
    General MLP class for checkpoint saving/loading
    Value, Critic and Actor networks inherit this class

    obs_dims: int
            observation space number of dimensions
    actions_dims: int
            action space number of dimensions

    takes an input of size obs_dims+action_dims
    """

    def __init__(self, output_sizes: Sequence[int],
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None) -> None:

        if non_linearity == 'relu':
            activation = jax.nn.relu
        else:
            raise NotImplemented

        super().__init__(output_sizes=output_sizes, activation=activation)
        # create checkpoint dir if it doesn't exist
        if chkpt_dir is not None:
            if not os.path.exists(chkpt_dir):
                os.mkdir(chkpt_dir)

        self.chkpt_dir = chkpt_dir
        self._chkpt_file = None  # will be set when saving
        self._input_dims = None  # should be set in child classes

    def assert_input_dim(self, x: chex.Array) -> None:
        assert x.shape[1] == self._input_dims, \
            "input dimension {x.shape} doesn't match expected dimension [batch_size, {self._input_dims}]"

    def save_checkpoint(self, file='value_net'):
        """call with a unique filename, otherwise previous file will be overwrited"""
        logging.info(f'Saving value network to {self.chkpt_dir}/{file}')
        self._chkpt_file = file
        with open(os.path.join(self.chkpt_dir, file), 'wb') as f:
            pickle.dump(self.params_dict(), f)

    def load_checkpoint(self, file=None):
        """loads last saved checkpoint by default"""
        logging.info(f'Loading value network from {self.chkpt_dir}/{file}')
        if file is None:
            file = self._chkpt_file
        with open(os.path.join(self.chkpt_dir, file), 'rb') as f:
            None
            # TODO check
            # self.params_dict() = pickle.load(f)


class CriticNetwork(CustomMLP):
    def __init__(self, obs_dims: int, action_dims: int,
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None):
        super().__init__(
            output_sizes=(256, 256, 1), non_linearity=non_linearity, chkpt_dir=chkpt_dir
        )

        self._input_dims = obs_dims + action_dims  # to assert correct input dims

    def __call__(self, state_action_pair: chex.Array) -> chex.Array:
        self.assert_input_dim(state_action_pair)
        return super().__call__(state_action_pair)


class ValueNetwork(CustomMLP):
    def __init__(self, obs_dims: int,
                 non_linearity: str = 'relu',
                 chkpt_dir: str = None):
        super().__init__(
            output_sizes=(256, 256, 1), non_linearity=non_linearity, chkpt_dir=chkpt_dir
        )

        self._input_dims = obs_dims  # to assert correct input dims

    def __call__(self, state: chex.Array) -> chex.Array:
        self.assert_input_dim(state)
        return super().__call__(state)


class ActorNetwork(CustomMLP):
    def __init__(self, obs_dims, action_dims,
                 non_linearity='relu',
                 chkpt_dir=None) -> None:
        super().__init__(
            output_sizes=(256, 256, 1), non_linearity=non_linearity, chkpt_dir=chkpt_dir
        )

        self.input_dims = obs_dims
        self.reparam_noise = 1e-6

    def __call__(x: chex.Array) -> chex.Array:
        # handle sampling from distribution
        pass
