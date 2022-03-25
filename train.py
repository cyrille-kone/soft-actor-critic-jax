import envs
from agents import SACAgent

if __name__ == '__main__':
    env = envs.PendulumEnv()
    n_trajectories = 200
    # checkpoint_file = 
    # plot file = 

    agent = SACAgent()


    best_reward = 0 # env.reward_spec().minimum() ?  # should be min reward possible
    all_rewards = []  # for future plotting

    train_every = 10
    save_every = 100

    for i in range(n_trajectories):
        timestep = env.reset()
        traj_reward = 0
        while timestep.step_type != 'LAST':
            action = agent.step(timestep)
            timestep_ = env.step(action)  # underscore denotes timestep t+1
            agent.record(timestep)
            traj_reward += timestep.reward

            if i % train_every == 0:
                agent.learner_step()

            if i % save_every == 0:
                agent.save_checkpoint(chkpt_dir='dir_save/', id=f'{i}')

            timestep = timestep_

        if traj_reward > best_reward:
            best_reward = traj_reward
            print(f"Trajectory\t{i}/{n_trajectories}")
            print(f"Reward \t {traj_reward}, Best \t {best_reward}")

