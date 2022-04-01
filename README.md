# Soft Actor-Critic
This is the repo for the project of implementing the algorithm from the paper <a href="https://arxiv.org/pdf/1801.01290.pdf"> Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor </a>

## Installation 

If you want to run the code on mujoco environments, make sure [mujoco](https://github.com/openai/mujoco-py) is installed on your computer.
Then run the following commands
```bash
git clone https://github.com/cyrille-kone/deep-rl
cd deep-rl
python3 -m venv envsac
source envsac/bin/activate
pip install --upgrade -r requirements.txt
```

## Run
The config file contains all the agent hyperparameters as well as other options
```bash
python train.py --config "configs/sac.yaml"
```

## Plot results
After training, you can plot results by running. If you have specified a different output file for plotting, you can replace the command argument by the name of your file.
```bash
python plot.py to_plot.log
```

## Original SAC Paper 
https://arxiv.org/abs/1801.01290
