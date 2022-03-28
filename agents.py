# coding=utf-8
"""
PyCharm Editor
@ git Team
"""
import abc
import acme
import jax
import jax.numpy as jnp
import chex
import haiku as hk
import optax  # for adam optimizer
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
                 batch_size=256,
                 lr=3e-4,  # same for all networks
                 hidden_output_dims = (256, 256),
                 gamma=0.99,
                 tau=0.005, # target smoothing coefficient,
                 target_update=1000,
                 chkpt_dir=None):

        obs_dims = obs_spec().shape[0]
        action_dims = action_spec().shape[0]
        self.memory = ReplayBuffer(obs_dims, action_dims)

        self.value = hk.without_apply_rng(hk.transform(lambda x:
                            ValueNetwork(obs_dims,
                                         hidden_output_dims=hidden_output_dims,
                                         chkpt_dir=chkpt_dir)(x)))
        self.value_target = hk.without_apply_rng(hk.transform(lambda x:
                            ValueNetwork(obs_dims,
                                         hidden_output_dims=hidden_output_dims,
                                         chkpt_dir=chkpt_dir)(x)))

        # we use two distinct Q networks (see end of 4.2 in the paper)
        self.Q1 = hk.without_apply_rng(hk.transform(lambda x:
                            CriticNetwork(obs_dims,
                                          action_dims,
                                          hidden_output_dims=hidden_output_dims,
                                          chkpt_dir=chkpt_dir)(x)))
        self.Q2 = hk.without_apply_rng(hk.transform(lambda x:
                            CriticNetwork(obs_dims,
                                          action_dims,
                                          hidden_output_dims=hidden_output_dims,
                                          chkpt_dir=chkpt_dir)(x)))
        self.actor = hk.without_apply_rng(hk.transform(lambda x:
                            ActorNetwork(obs_dims,
                                         action_dims,
                                         hidden_output_dims=hidden_output_dims,
                                         chkpt_dir=chkpt_dir)(x)))

        self.rng = rng

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.batch_size = batch_size
        self.gamma = gamma

        # initialize networks parameters
        self.rng, value_key, Q1_key, Q2_key, actor_key = jax.random.split(self.rng, 4)

        dummy_obs = jnp.ones(obs_spec().shape)
        dummy_action = jnp.ones(action_spec().shape)
        self.value_params = self.value_target_params = self.value.init(value_key, dummy_obs)
        self.Q1_params = self.Q1.init(Q1_key, jnp.concatenate((dummy_obs, dummy_action)))
        self.Q2_params = self.Q2.init(Q2_key, jnp.concatenate((dummy_obs, dummy_action)))
        self.actor_params = self.actor.init(value_key, dummy_obs)

        # create and initialize optimizers
        self.value_opt = optax.adam(lr)
        self.Q1_opt = optax.adam(lr)
        self.Q2_opt = optax.adam(lr)
        self.actor_opt = optax.adam(lr)

        self.value_opt_state = self.value_opt.init(self.value_params)  # same optimizer for value and value_target
        self.Q1_opt_state = self.Q1_opt.init(self.Q1_params)
        self.Q2_opt_state = self.Q2_opt.init(self.Q2_params)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)


    # is this supposed to be batched_actor_step ?
    def select_actions(self, observations: chex.Array) -> chex.Array:
        """
        observations: chex.Array
                batch of states

        for each observations, generates action distribution from actor
        and samples from this distribution
        also computes log_probs, to compute loss function
        return: actions, log_probs
        """
        self.rng, key = jax.random.split(self.rng, 2)
        mus, log_sigmas = self.actor.apply(self.actor_params, observations)

        # see appendix C in SAC paper

        # sample actions according to normal distributions
        actions = jax.random.multivariate_normal(key, mus, jnp.array([jnp.diag(jnp.exp(s)) for s in log_sigmas]))

        # compute log_likelihood of the sampled actions
        log_probs = -0.5*jnp.log(2*jnp.pi) - log_sigmas - ((actions-mus)/(2*jnp.exp(log_sigmas)))**2

        # squash actions to enforce action bounds
        actions = jnp.tanh(actions) * (self.action_spec().maximum - self.action_spec().minimum)

        # compute squashed log-likelihood
        # ! other implementations put a relu in the log
        # + 1e-6 to prevent log(0)
        log_probs -= jnp.sum(jnp.log(1 - jnp.tanh(actions)**2 + 1e-6), axis=1)


        return actions, log_probs

    def select_action(self, obs: chex.Array) -> chex.Array:
        """
        obs: chex.Array
                a single observation
        returns a single action sampled from actor's  distribution

        This is meant to be used while interacting with the environment
        """
        action, _ = self.select_actions(jnp.expand_dims(obs, 0))
        return action.squeeze(axis=0)

    def record(self, t, action, t_):
        self.store_transition(t.observation, action, t_.reward, t_.observation, t_.step_type=='LAST')

    # TODO
    def store_transition(self, state, action, reward, next_state, done):
        """strores transition in replay buffer"""
        pass

    def update_target_network(self, tau=None):
        """soft update of network parameters"""
        pass

    def save_checkpoint(chkpt_dir):
        """uses networks save_checkpoint methods"""
        pass

    def load_checkpoint(chkpt_dir):
        pass

    def learner_step(self):
        # get batch of transitions
        self.rng, key = jax.random.split(self.rng, 2)
        batch = self.memory.sample_batch(self.batch_size, key)

        # compute actions and log_probs
        actions, log_probs = self.select_actions(batch.state)

        # get minimum Q value (see end of 4.2 in the paper)
        state_action_input = jnp.concatenate(batch.state, batch.actions, axis=1)
        q1 = jax.lax.stop_gradient(self.Q1.apply(self.Q1_params, state_action_input))
        q2 = jax.lax.stop_gradient(self.Q2.apply(self.Q2_params, state_action_input))
        q = jnp.minimum(q1, q2)

        # get q_hat (see eq. 8 in paper)
        q_hat = batch.reward + (1-batch.done)*self.gamma\
                self.value_target.apply(self.value_target_params, batch.next_state)

        # compute value, q and
        value = self.value.apply(self.value_params, batch.state)
        critic = self.critic.apply(self.critic_params, jnp.concatenate(batch.state, batch.actions, axis=1))

        # update agents

        # TODO: implement self.value_loss_fn
        value_grads = jax.grads(self.value_loss_fn)(self.value_params, batch.state)
        value_updates, self.value_opt_state = self.value_opt.update(value_grads, self.value_opt_state)
        self.value_params = optax.apply_updates(self.value_params, value_updates)

        # TODO: implement self.Q_loss_fn
        q1_grads = jax.grad(self.Q_loss_fn)(self.Q1_params, state_action_input)
        q1_updates, self.Q1_opt_state = self.Q1_opt.update(Q1_grads, self.Q1_opt_state)
        self.Q1_params = optax.apply_updates(self.Q1_params, Q1_updates)

        q2_grads = jax.grad(self.Q_loss_fn)(self.Q2_params, state_action_input)
        q2_updates, self.Q2_opt_state = self.Q2_opt.update(Q2_grads, self.Q2_opt_state)
        self.Q2_params = optax.apply_updates(self.Q2_params, Q2_updates)

        # TODO: implement self.actor_loss_fn
        actor_grads = jax.grads(self.actor_loss_fn)(self.actor_params, batch.state)
        actor_updates, self.actor_opt_state = self.actor_opt.update(actor_grads, self.actor_opt_state)
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)


        return 0


