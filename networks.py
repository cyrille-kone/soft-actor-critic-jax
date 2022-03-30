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


# Inherits from MLP so no need to reimplement forward/__call__
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
        # create checkpoint dir if it doesn't exist
        if chkpt_dir is not None:
            if not os.path.exists(chkpt_dir):
                os.mkdir(chkpt_dir)

        self.chkpt_dir = chkpt_dir
        self.chkpt_file = None           # will be set when saving

        self._expected_input_dims = None  # should be set in child classes

    def __call__(self, x: chex.Array) -> chex.Array:
        assert x.shape[-1] == self._expected_input_dims,\
        f"input dimension {x.shape} doesn't match expected dimension [batch_size, {self._expected_input_dims}]"
        return super().__call__(x)

    def save_checkpoint(self, file=None):
        """call with a unique filename, otherwise previous file will be overwrited"""
        logger.info(f'Saving value network to {self.chkpt_dir}/{file}')
        if file is not None:
            self.chkpt_file = file
        with open(os.path.join(self.chkpt_dir, self.chkpt_file), 'wb') as f:
            pickle.dump(self.params_dict(), f)

    def load_checkpoint(self, file=None):
        """loads last saved checkpoint by default"""
        logger.info(f'Loading value network from {self.chkpt_dir}/{file}')
        if file is not None:
            self.chkpt_file = file

        if not os.path.exists(os.path.join(self.chkpt_dir, self.chkpt_file)):
            logger.warning('could not load checkpoint file (file not found)')
            return None

        with open(os.path.join(self.chkpt_dir, self.chkpt_file), 'rb') as f:
            print(self.params_dict)
            self.params_dict = pickle.load(f)


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

        self._expected_input_dims = obs_dims
        self.reparam_noise = 1e-6

    def __call__(self, state: chex.Array) -> chex.Array:
        h = super().__call__(state)
        mu, log_sigma = jnp.split(h, 2, axis=-1)

        # TODO: these next lines are present in other implementations, 
        # but i don't see where they are explained in the paper ?

        log_sigma = jnp.clip(log_sigma, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
        # log_sigma = jnp.clip(log_sigma, min=jnp.log(self.reparam_noise))  # prevent -inf values
        return mu, log_sigma  # need to reshape that ?

