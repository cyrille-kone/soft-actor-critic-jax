# coding=utf-8
r"""
PyCharm Editor
Author @git Team
"""
import jax
import unittest
import replay_buffer

seed = 0


class TestReplayBuffer(unittest.TestCase):
    def test_replay_buffer(self):
        print("creating replay buffer...")
        buffer = replay_buffer.ReplayBuffer(obs_shape=(3,), action_shape=(5,), max_length=10)
        key = jax.random.PRNGKey(seed)
        print("storing dummy data...")
        for i in range(12):
            key, rng = jax.random.split(key)
            obs = jax.random.uniform(rng, shape=(3,))
            action = jax.random.uniform(rng, shape=(5,))
            reward = jax.random.uniform(rng, shape=(1,))
            next_state = jax.random.uniform(rng, shape=(3,))
            done = jax.random.uniform(rng, shape=(1,))
            buffer.store_transition(obs, action, reward, next_state, done)
            if i == 0:
                first_obs = obs
            if i == buffer.max_length:
                assert buffer._memory[0] != first_obs
        print("sampling transition batch")
        key, rng = jax.random.split(key)
        batch = buffer.sample_batch(400, rng)
        self.assertEqual(batch.state.shape, (400, 3), "Replay buffer sampled batch checking failed")
        # print(batch.state.shape)
        # print(batch)


if __name__ == "__main__":
    unittest.main()
