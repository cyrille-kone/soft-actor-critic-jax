import argparse
import base64
from pathlib import Path
from envs import *
import jax
import jax.numpy as jnp
import sys
from agents import SACAgent, RandomAgent
from dm_env import StepType
import yaml
import IPython
from utils import logger


def evaluate(environment, agent, evaluation_episodes, max_steps):
  frames = []
  avg_rew = 0.
  for episode in range(evaluation_episodes):
    timestep = environment.reset()
    episode_return = 0
    steps = 0
    while not timestep.last() and steps < max_steps:

      action = agent.select_action(timestep.observation, deterministic=True)
      timestep = environment.step(action)
      steps += 1
      episode_return += timestep.reward
    avg_rew += episode_return
  avg_rew /= evaluation_episodes
  print(
      f'Evaluation ended with reward {avg_rew} in {evaluation_episodes} episodes'
  )
  return 0

def display_video(frames, filename='temp.mp4', frame_repeat=4):
  """Save and display video."""
  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      for _ in range(frame_repeat):
        video.append_data(frame)
  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
  return IPython.display.HTML(video_tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/example.yaml', type=str)
    args = parser.parse_args()
    with open(Path(args.config), 'r') as f:
        config_args = yaml.safe_load(f.read())

    n_trajectories = config_args['n_trajectories']
    seed = config_args['seed']

    env = eval(config_args['env'])(for_evaluation=False)

    print('Observation shape : ', (env.observation_spec().shape[0]))
    print('Action shape : ', (env.action_spec().shape[0]))
    print('Parameters : ', config_args)
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    logger.info("Creating SAC agent")
    agent = SACAgent(rng, env.observation_spec, env.action_spec, jit=False, **config_args['agent_kwargs'])
    # agent = RandomAgent(rng, env.observation_spec, env.action_spec)

    best_reward = 0   # env.reward_spec().minimum() ?  # should be min reward possible
    all_rewards = []  # for future plotting
    max_steps = config_args['max_steps_per_episode']   # maximum number of steps in a trajectory to prevent infinite loops

    train_every = config_args['train_interval']
    save_every = config_args['save_interval']
    eval_every = config_args['eval_interval']

    logger.info('Beginning training')
    for i in range(1, n_trajectories+1):
        logger.debug('trajectory ' + f'{i}')
        timestep = env.reset()
        traj_reward = 0
        n_steps = 0
        while timestep.step_type != StepType.LAST and n_steps <= max_steps:
            logger.debug('step ' + f'{n_steps}')
            action = agent.select_action(timestep.observation, logging=True)
            logger.debug('action ' + f'{round(action[0], 3)}')
            timestep_ = env.step(action)  # underscore denotes timestep t+1
            agent.record(timestep, action, timestep_)
            traj_reward += timestep_.reward
            timestep = timestep_
            n_steps += 1

            if n_steps % train_every == 0: # and len(agent.memory) >= agent.batch_size:
                agent.learner_step()
                logger.debug('value ' + f'{agent.value.apply(agent.value_params, timestep.observation)[0]}')
                logger.debug('q ' + f"{agent.Q.apply(agent.Q1_params, jnp.concatenate((timestep.observation, action), axis=-1))[0]}")

        logger.debug('trajectory_reward ' + f'{traj_reward}')
        logger.info(f"Trajectory\t{i}/{n_trajectories}")
        logger.info(f"Reward \t {traj_reward}, Best \t {best_reward}")
        if i % eval_every == 0:
            evaluate(env, agent, evaluation_episodes=5, max_steps=max_steps)

        if i % save_every == 0:
            agent.save_checkpoint('dir_save')

        if traj_reward > best_reward:
            best_reward = traj_reward

