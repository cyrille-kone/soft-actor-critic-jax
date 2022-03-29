# coding=utf-8
"""
PyCharm Editor
@ git Team
"""
import abc
from functools import partial
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
from utils import mse_loss


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
                 actor_lr=3e-4,
                 value_lr=3e-4,
                 q_lr=3e-4,
                 hidden_dims = (256, 256),
                 discount=0.99,
                 tau=0.005, # target smoothing coefficient,
                 target_period_update=1,
                 chkpt_dir=None):

        obs_dims = obs_spec().shape[0]
        action_dims = action_spec().shape[0]
        self.memory = ReplayBuffer(obs_dims, action_dims)

        print("initializing networks")
        self.value = hk.without_apply_rng(hk.transform(lambda x:
                            ValueNetwork(obs_dims,
                                         hidden_output_dims=hidden_dims,
                                         chkpt_dir=chkpt_dir)(x)))

        # we use two distinct Q networks (see end of 4.2 in the paper)
        self.Q = hk.without_apply_rng(hk.transform(lambda x:
                            CriticNetwork(obs_dims,
                                          action_dims,
                                          hidden_output_dims=hidden_dims,
                                          chkpt_dir=chkpt_dir)(x)))
        self.actor = hk.without_apply_rng(hk.transform(lambda x:
                            ActorNetwork(obs_dims,
                                         action_dims,
                                         hidden_output_dims=hidden_dims,
                                         chkpt_dir=chkpt_dir)(x)))

        self.rng = rng

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

        # initialize networks parameters
        self.rng, value_key, Q1_key, Q2_key, actor_key = jax.random.split(self.rng, 5)

        dummy_obs = jnp.ones(obs_spec().shape)
        dummy_action = jnp.ones(action_spec().shape)
        self.value_params = self.value_target_params = self.value.init(value_key, dummy_obs)
        self.Q1_params = self.Q.init(Q1_key, jnp.concatenate((dummy_obs, dummy_action)))
        self.Q2_params = self.Q.init(Q2_key, jnp.concatenate((dummy_obs, dummy_action)))
        self.actor_params = self.actor.init(actor_key, dummy_obs)

        # create and initialize optimizers
        self.value_opt = optax.adam(value_lr)
        self.Q1_opt = optax.adam(q_lr)
        self.Q2_opt = optax.adam(q_lr)
        self.actor_opt = optax.adam(actor_lr)

        self.value_opt_state = self.value_opt.init(self.value_params)  # same optimizer for value and value_target
        self.Q1_opt_state = self.Q1_opt.init(self.Q1_params)
        self.Q2_opt_state = self.Q2_opt.init(self.Q2_params)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)

        self.update_value = jax.jit(self._update_value)
        self.update_actor = jax.jit(self._update_actor)
        self.update_q = jax.jit(self._update_q)

    # is this supposed to be batched_actor_step ?
    def select_actions(self, actor_params, observations: chex.Array) -> chex.Array:
        """
        observations: chex.Array
                batch of states

        for each observations, generates action distribution from actor
        and samples from this distribution
        also computes log_probs, to compute loss function
        return: actions, log_probs
        """
        self.rng, key = jax.random.split(self.rng, 2)
        mus, log_sigmas = self.actor.apply(actor_params, observations)

        # see appendix C in SAC paper

        # sample actions according to normal distributions
        actions = mus + jax.random.normal(key, mus.shape) * jnp.exp(log_sigmas)

        # compute log_likelihood of the sampled actions
        log_probs = -0.5*jnp.log(2*jnp.pi) - log_sigmas - ((actions-mus)/(2*jnp.exp(log_sigmas)))**2

        # compute squashed log-likelihood
        # ! other implementations put a relu in the log
        # + 1e-6 to prevent log(0)
        log_probs -= jnp.sum(
            jnp.log(jax.nn.relu(1 - jnp.tanh(actions)**2) + 1e-6),
            axis=1, keepdims=True
        )

        # squash actions to enforce action bounds
        actions = jnp.tanh(actions) * self.action_spec().maximum

        return actions, log_probs

    def select_action(self, obs: chex.Array) -> chex.Array:
        """
        obs: chex.Array
                a single observation
        returns a single action sampled from actor's  distribution

        This is meant to be used while interacting with the environment
        """
        action, _ = self.select_actions(self.actor_params, jnp.expand_dims(obs, 0))
        return action.squeeze(axis=0)

    def record(self, t, action, t_):
        self.memory.store_transition(t.observation, action, t_.reward, t_.observation, t_.step_type)

    def update_target_network(self, ):
        """soft update of network parameters"""
        pass
    
    # @partial(jax.jit, static_argnums=(0,))
    # def update_value_network(self, value, soft_value):
    #     value_grads = jax.grad(self.value_loss)(self.value_params, batch, q-log_probs)
    #     value_updates, self.value_opt_state = self.value_opt.update(value_grads, self.value_opt_state)
    #     self.value_params = optax.apply_updates(self.value_params, value_updates)
    
    # def update_actor(self, log_probs, q):
    #     actor_loss, actor_grads = jax.value_and_grad(self.actor_loss)(self.actor_params, log_probs, q)
    #     actor_updates, self.actor_opt_state = self.actor_opt.update(actor_grads, self.actor_opt_state)
    #     self.actor_params = optax.apply_updates(self.actor_params, actor_updates)
    
    # def update_critic(self, q1_r, q_hat):
    #     q1_grads = jax.grad(self.critic_loss)(self.Q1_params, q1_r, q_hat)
    #     q1_updates, self.Q1_opt_state = self.Q1_opt.update(q1_grads, self.Q1_opt_state)
    #     self.Q1_params = optax.apply_updates(self.Q1_params, q1_updates)

    def save_checkpoint(self, chkpt_dir):
        """uses networks save_checkpoint methods"""
        pass

    def load_checkpoint(self, chkpt_dir):
        pass

    def _update_value(self, value_params, value_opt_state, state, q, log_probs):

        def value_loss_fn(value_params, state, target):
            value = self.value.apply(value_params, state)
            return 0.5*mse_loss(value, target)
        
        value_grads = jax.grad(value_loss_fn)(value_params, state, q-log_probs)
        value_updates, value_opt_state = self.value_opt.update(value_grads, value_opt_state)
        value_params = optax.apply_updates(value_params, value_updates)
        return value_params, value_opt_state

    def _update_actor(self, actor_params, actor_opt_state, q, observations):

        def actor_loss_fn(actor_params, q, observations):
            _, log_probs = self.select_actions(actor_params, observations)
            return (log_probs - q).mean()
        
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)( 
            actor_params, q, observations
        )
        actor_updates, actor_opt_state = self.actor_opt.update(actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)
        return actor_loss, actor_params, actor_opt_state

    def _update_q(self, q1_params, q1_opt_state, q2_params, q2_opt_state, batch):
        q_hat = batch.reward * 20 + (1-batch.done)*self.discount*\
                self.value.apply(self.value_target_params, batch.next_state)
        def q_loss(q_params, q_hat, state, action):
            state_action_input = jnp.concatenate((state, action), axis=1)
            q_r = self.Q.apply(q_params, state_action_input)  # _r is for replay buffer
            return 0.5*mse_loss(q_r, q_hat)
        
        q1_grads = jax.grad(q_loss)(q1_params, q_hat, batch.state, batch.action)
        q1_updates, q1_opt_state = self.Q1_opt.update(q1_grads, q1_opt_state)
        q1_params = optax.apply_updates(q1_params, q1_updates)

        q2_grads = jax.grad(q_loss)(q2_params, q_hat, batch.state, batch.action)
        q2_updates, q2_opt_state = self.Q2_opt.update(q2_grads, q2_opt_state)
        q2_params = optax.apply_updates(q2_params, q2_updates)

        return q1_params, q1_opt_state, q2_params, q2_opt_state

    def learner_step(self):
        # get batch of transitions
        self.rng, key = jax.random.split(self.rng, 2)
        batch = self.memory.sample_batch(self.batch_size, key)

        # compute actions and log_probs from current policy
        actions, log_probs = self.select_actions(self.actor_params, batch.state)

        # get minimum Q value (see end of 4.2 in the paper)
        # use actions from current policy and not from replay buffer (see 4.2 just after eq.6)
        state_action_input = jnp.concatenate((batch.state, actions), axis=1)
        q1 = jax.lax.stop_gradient(self.Q.apply(self.Q1_params, state_action_input))
        q2 = jax.lax.stop_gradient(self.Q.apply(self.Q2_params, state_action_input))
        q = jnp.minimum(q1, q2)

        # compute value
        # value = self.value.apply(self.value_params, batch.state)

        # compute value gradients and update value network
        self.value_params, self.value_opt_state = self.update_value(
            self.value_params, self.value_opt_state,
            batch.state, q, log_probs
        )
        # !! I didn't understand what the "reparameterization trick" was so I didn't code it yet

        # compute actor gradients and update actor network
        actor_loss, self.actor_params, self.actor_opt_state = self.update_actor(
            self.actor_params, self.actor_opt_state,
            key, q, batch.state
        )

        # get q_hat (see eq. 8 in paper)
        # q_hat = batch.reward + (1-batch.done)*self.discount*\
        #         self.value_target.apply(self.value_target_params, batch.next_state)

        # compute q (use actions from replay buffer this time (paper eq. 7))
        # state_action_input = jnp.concatenate((batch.state, batch.action), axis=1)
        # q1_r = self.Q1.apply(self.Q1_params, state_action_input)  # _r is for replay buffer
        # q2_r = self.Q2.apply(self.Q2_params, state_action_input)

        # compute q gradients and update critic networks
        self.Q1_params, self.Q1_opt_state, self.Q2_params, self.Q2_opt_state = self.update_q(
            self.Q1_params, self.Q1_opt_state,
            self.Q2_params, self.Q2_opt_state,
            batch
        )

        # soft-update target value network parameters 
        # jax.tree_multimap applies update recursively to weight and biases of the layers of the networks
        self.value_target_params = jax.tree_multimap(
            lambda params, target_params: self.tau*params + (1-self.tau)*target_params,
            self.value_params, self.value_target_params)

        return actor_loss

