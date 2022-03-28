from replay_buffer import ReplayBuffer
import jax

seed = 0
# [cyrk] J'ai plagié :) le test dans tests/test_replay_buffer

def test_replay_buffer():
    print("creating replay buffer...")
    buffer = ReplayBuffer(obs_shape=(3,), action_shape=(5,), max_length=10)
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
    print(batch.state.shape)
    print(batch)


if __name__ == "__main__":
    test_replay_buffer()
