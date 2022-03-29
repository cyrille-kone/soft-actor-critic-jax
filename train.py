from envs import *
import jax
import sys
from agents import SACAgent
from dm_env import StepType
import yaml

if __name__ == '__main__':
    # checkpoint_file = 
    # plot file = 
    with open('./configs/example.yaml', 'r') as f:
        config_args = yaml.safe_load(f.read())
    env = eval(config_args['env'])(for_evaluation=False)
    n_trajectories = config_args['n_trajectories']
    seed = config_args['seed']
    rng = jax.random.PRNGKey(seed)

    agent = SACAgent(rng, env.observation_spec, env.action_spec, **config_args['agent_kwargs'])


    best_reward = 0   # env.reward_spec().minimum() ?  # should be min reward possible
    all_rewards = []  # for future plotting
    max_steps = config_args['max_steps_per_episode']   # maximum number of steps in a trajectory to prevent infinite loops

    train_every = config_args['train_interval']
    save_every = config_args['save_interval']

    for i in range(1, n_trajectories+1):
        print(f'trajectory {i}')
        timestep = env.reset()
        traj_reward = 0
        n_steps = 0
        while timestep.step_type != StepType.LAST and n_steps <= max_steps:
            action = agent.select_action(timestep.observation)
            timestep_ = env.step(action)  # underscore denotes timestep t+1
            agent.record(timestep, action, timestep_)
            traj_reward += timestep_.reward
            timestep = timestep_
            n_steps += 1

            if n_steps % train_every == 0:
                actor_loss = agent.learner_step()

        print(traj_reward)
        if i % save_every == 0:
            agent.save_checkpoint('dir_save')

        if i % save_every == 0:
            agent.save_checkpoint('dir_save')

        if traj_reward > best_reward:
            best_reward = traj_reward
            print("Trajectory\t{i}/{n_trajectories}")
            print("Reward \t {traj_reward}, Best \t {best_reward}")

