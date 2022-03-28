# TODO: import agents

if __name__ == '__main__':
    # make environment env =
    n_trajectories = 200
    # checkpoint_file = 
    # plot file =
    best_reward = 0 # env.reward_spec().minimum() ?  # should be min reward possible
    load_checkpoint = False  # TODO: make that a command option ? | maybe we should write a function for that [cyrk]

    train_every = 10

    for i in range(n_trajectories):
        timestep = env.reset()
        traj_reward = 0
        while timestep.step_type != 'LAST':
            action = agent.step(timestep)
            timestep_ = env.step(action)  # underscore denotes timestep t+1
            agent.record(timestep)
            traj_reward += timestep.reward

            if i % train_every == 0:
                agent.train()

            timestep = timestep_

        if traj_reward > best_reward:
            best_reward = traj_reward
            print(f"Trajectory\t{i}/{n_trajectories}")
            print(f"Reward \t {traj_reward}, Best \t {best_reward}")
