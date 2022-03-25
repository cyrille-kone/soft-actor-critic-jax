import jax
import jax.numpy as jnp
from collections import namedtuple

Transition = namedtuple('Transition', 'state action reward next_state done')


class ReplayBuffer():
    """A simple replay buffer class to keep track of the agent's past experience"""

    def __init__(self, obs_shape, action_shape, max_length=10_000):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_length = max_length

        self._memory = []

    def store_transition(self, state, action, reward, next_state, done) -> None:
        while len(self._memory) >= self.max_length:
            del self._memory[0]
        self._memory.append(Transition(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       done=done))

    def sample_batch(self, batch_size: int, rng: jnp.ndarray) -> Transition:
        """Returns a Transition of batches"""
        indices = jax.random.randint(rng, (batch_size,), 0, len(self._memory))

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for i in indices:
            t = self._memory[i]
            states.append(t.state)
            actions.append(t.action)
            next_states.append(t.next_state)
            rewards.append(t.reward)
            dones.append(t.done)

        return Transition(
            jnp.array(states),
            jnp.array(actions),
            jnp.array(next_states),
            jnp.array(rewards),
            jnp.array(dones)
        )
