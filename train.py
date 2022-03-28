import envs
import jax
from agents import SACAgent
from dm_env import StepType
import yaml

if __name__ == '__main__':
    env = envs.PendulumEnv(for_evaluation=False)
    n_trajectories = 2
    # checkpoint_file = 
    # plot file = 
    with open('./configs/example.yaml', 'r') as f:
        config_args = yaml.safe_load(f.read())
    print(config_args['env'])
    seed = 0
    rng = jax.random.PRNGKey(seed)

    agent = SACAgent(rng, env.observation_spec, env.action_spec)


    best_reward = 0 # env.reward_spec().minimum() ?  # should be min reward possible
    all_rewards = []  # for future plotting

    train_every = 10
    save_every = 100

    for i in range(1, n_trajectories+1):
        timestep = env.reset()
        traj_reward = 0
        while timestep.step_type != StepType.LAST:
            action = agent.select_action(timestep.observation)
            timestep_ = env.step(action)  # underscore denotes timestep t+1
            agent.record(timestep, action, timestep_)
            traj_reward += timestep_.reward

            if i % train_every == 0:
                agent.learner_step()

            if i % save_every == 0:
                agent.save_checkpoint(chkpt_dir='dir_save/', id='{i}')

            timestep = timestep_

        if traj_reward > best_reward:
            best_reward = traj_reward
            print("Trajectory\t{i}/{n_trajectories}")
            print("Reward \t {traj_reward}, Best \t {best_reward}")

