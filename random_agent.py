import jax

class RandomAgent():
    def __init__(self,
                 rng,
                 obs_spec,
                 action_spec):
        self.rng = rng
        self.action_spec = action_spec
        self.memory = [0]
        self.batch_size = 0

    def select_action(self, obs, deterministic=False):
        self.rng, key = jax.random.split(self.rng, 2)
        return jax.random.uniform(key, self.action_spec().shape)

    def learner_step(self):
        return None

    def record(self, a, b, c):
        pass

