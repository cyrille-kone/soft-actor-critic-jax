from replay_buffer import ReplayBuffer
import envs
import jax

seed = 0

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

        if i==0:
            first_obs = obs
        if i==buffer.max_length:
            assert buffer._memory[0] != first_obs

    print("sampling transition batch")
    key, rng = jax.random.split(key)
    batch = buffer.sample_batch(400, rng)
    print(batch.state.shape)
    print(batch)

def test_environments():
    print("creating the environments...")
    pendulum_env = envs.PendulumEnv(for_evaluation=False)
    reacher_env = envs.ReacherEnv(for_evaluation=True)
    invertedpendulum_env = envs.InvertedPendulumEnv(for_evaluation=False)


if __name__=="__main__":
    #test_replay_buffer()
    test_environments()
