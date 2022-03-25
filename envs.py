coding=utf-8
r"""
PyCharm Editor
@ git Team
"""

import gym
import chex
import dm_env
import mujoco_py
import logging
import imageio
import numpy as np
from acme import specs
from typing import Optional

# configure logger for the module
logger = logging.getLogger(__name__)
# create console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class MixinEnv(dm_env.Environment):
    def __init__(self, for_evaluation):
        self._for_evaluation = for_evaluation
        if self._for_evaluation:
            self.screen = []

    def close(self) -> None:
        if hasattr(self, "_env"):
            self._env.close()

    def save_video(self, filename, frame_repeat):
        if not hasattr(self, "screen"):
            logger.warning("No screen set for this environment")
        # if the frame list is empty
        if not self.screen:
            # no frame saved
            logger.error("No frames saved for this environment")
        with imageio.get_writer(filename, fps=60) as video:
            for frame in self.screen:
                for _ in range(frame_repeat):
                    video.append_data(frame)
            # Read video and display the video
        video = open(filename, 'rb').read()
        return video # TODO essayer pour voir si ça vaut la peine de retourner video



    def reset(self) -> dm_env.TimeStep:
        obs = self._env.reset()
        if self._for_evaluation:
            self.screen.append(self._env.render(mode='rgb_array'))
        return dm_env.restart(obs)


class PendulumEnv(MixinEnv):
    def __init__(self, for_evaluation: bool) -> None:
        super().__init__(for_evaluation)
        self._env = gym.make('Pendulum-v0')
        self._for_evaluation = for_evaluation
        if self._for_evaluation:
            self.screens = []

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        new_obs, reward, done, _ = self._env.step(action)
        if self._for_evaluation:
            self.screens.append(self._env.render(mode='rgb_array'))
        if done:
            return dm_env.termination(reward, new_obs)
        return dm_env.transition(reward, new_obs)

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(3,), minimum=-8., maximum=8., dtype=np.float32)

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(1,), minimum=-2., maximum=2., dtype=np.float32)


class PusherEnv(MixinEnv):
    def __init__(self, seed: int, for_evaluation: bool, evaluation_video_path: Optional[str] = None,
                 deltat: float = .05, max_steps: int = 100) -> None:
        super().__init__(for_evaluation)
        self._state = None
        self._deltat = deltat
        self._nb_steps = 0
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed=seed)
        self._nb_episodes = 0

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(1,), dtype=np.float32, minimum=-1., maximum=1.)

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=[-1, -np.inf], maximum=[1., np.inf])

    def reset(self) -> dm_env.TimeStep:
        self._nb_episodes += 1
        rnd = self._rng.uniform(low=-1., high=1.)
        x = np.sign(rnd).astype(np.float32)
        dx_dt = 0.
        self._state = np.array([x, dx_dt])
        self._nb_steps = 1
        return dm_env.restart(observation=self._state)

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        action = np.clip(action, a_min=-1, a_max=1)[0]
        self._nb_steps += 1
        if self._nb_steps > self._max_steps:
            return dm_env.termination(0., self._state)
        x, dx_dt = self._state
        new_x = max(min(x + self._deltat * dx_dt, 1.), -1.)
        hit_left_wall = float(new_x == -1.)
        hit_right_wall = float(new_x == 1.)
        new_dx_dt = dx_dt + action * self._deltat
        new_dx_dt = max(new_dx_dt, 0) * (1. - hit_right_wall) + min(new_dx_dt, 0.) * (1. - hit_left_wall)
        reward = -new_x ** 2
        self._state = np.array([new_x, new_dx_dt])
        return dm_env.transition(reward, self._state)

    def close(self) -> None:
        pass


class InvertedPendulumEnv(MixinEnv):
    def __init__(self, for_evaluation: bool) -> None:
        super().__init__(for_evaluation)
        self._env = gym.make('InvertedPendulum-v2')

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        new_obs, reward, done, _ = self._env.step(action)
        if self._for_evaluation:
            self.screen.append(self._env.render(mode='rgb_array'))
        if done:
            return dm_env.termination(reward, new_obs)
        return dm_env.transition(reward, new_obs)

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(4,), minimum=-np.inf, maximum=np.inf, dtype=np.float32)

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(1,), minimum=-3., maximum=3., dtype=np.float32)


class ReacherEnv(MixinEnv):
    def __init__(self, for_evaluation: bool) -> None:
        super().__init__(for_evaluation)
        self._env = gym.make('Reacher-v2')

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        new_obs, reward, done, _ = self._env.step(action)
        if self._for_evaluation:
            self.screen.append(self._env.render(mode='rgb_array'))
        if done:
            return dm_env.termination(reward, new_obs)
        return dm_env.transition(reward, new_obs)

    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(11,), minimum=-np.inf, maximum=np.inf, dtype=np.float32)

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(2,), minimum=-1., maximum=1., dtype=np.float32)
