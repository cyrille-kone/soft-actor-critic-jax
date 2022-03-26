# coding=utf-8
"""
PyCharm Editor
@ git Team
"""
import abc
import acme
import chex
from typing import Mapping

from new_types import Trajectory
from networks import ValueNetwork, CriticNetwork, ActorNetwork
from replay_buffer import ReplayBuffer


# A very simple agent API, with just enough to interact with the environment
# and to update its potential parameters.
class Agent(object):
    @abc.abstractmethod
    def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
        """One step of learning on a trajectory.

        The mapping returned can contain various logs.
        """
        pass

    @abc.abstractmethod
    def batched_actor_step(self, observations: acme.types.NestedArray) -> acme.types.NestedArray:
        """Returns actions in response to observations.

        Observations are assumed to be batched, i.e. they are typically arrays, or
        nests (think nested dictionaries) of arrays with shape (B, F_1, F_2, ...)
        where B is the batch size, and F_1, F_2, ... are feature dimensions.
        """
        pass


class SACAgent(Agent):
    def __init__(self,
                 rng,
                 obs_spec,
                 action_spec,
                 lr=3e-4,  # same for all networks
                 hidden_output_dims = (256, 256),
                 gamma=0.99,
                 tau=0.005, # target smoothing coefficient,
                 target_update=1000,
                 chkpt_dir=None):

        obs_dims = obs_spec().shape[-1]
        action_dims = action_spec().shape[-1]
        self.memory = ReplayBuffer(obs_dims, action_dims)

        self.value_target = ValueNetwork(obs_dims,
                                         hidden_output_dims=hidden_output_dims,
                                         chkpt_dir=chkpt_dir)
        self.value = ValueNetwork(obs_dims,
                                  hidden_output_dims=hidden_output_dims,
                                  chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(obs_dims,
                                    action_dims,
                                    hidden_output_dims=hidden_output_dims,
                                    chkpt_dir=chkpt_dir)
        self.actor = ActorNetwork(obs_dims,
                                  action_dims,
                                  hidden_output_dims=hidden_output_dims,
                                  chkpt_dir=chkpt_dir)

        self.rng = rng
        self.obs_spec = obs_spec
        self.action_spec = action_spec


    def select_actions(observations: chex.Array) -> chex.Array:
        """
        observations: chex.Array
                batch of states

        for each observations, generates action distribution from actor
        and samples from this distribution
        also computes log_probs, to compute loss function
        return: actions, log_probs
        """
        rng, key = jax.random.split(rng, 2)
        mus, log_sigmas = self.actor(observations)
        # sample actions according to normal distributions
        actions = jax.random.multivariate_normal(key, mus, jnp.diag(jnp.exp(sigmas)))

        # compute log_likelihood of the sampled actions
        log_probs = -0.5*jnp.log(2*jnp.pi) - log_sigmas - ((actions-mus)/(2*jnp.exp(sigmas)))**2

        # squash actions to enforce action bounds
        actions = jnp.tanh(actions) * (self.action_spec().maximum() - self.action_spec().minimum())

        # compute squashed log-likelihood
        # ! other implementations put a relu in the log
        # + 1e-6 to prevent log(0)
        log_probs -= jnp.sum(jnp.log(1 - jnp.tanh(actions)**2 + 1e-6), axis=1)


        return actions, log_probs

    # TODO
    def store_transition(self, state, action, reward, next_state, done):
        pass

    def update_target_network(self, tau=None):
        """soft update of network parameters"""
        pass

    def save_checkpoint(chkpt_dir):
        pass

    def load_checkpoint(chkpt_dir):
        pass

    def learner_step():
        pass
