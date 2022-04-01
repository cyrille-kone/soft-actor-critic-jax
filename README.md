# Soft Actor-Critic
This is the repo for the project of implementing the algorithm from the paper <a href="https://arxiv.org/pdf/1801.01290.pdf"> Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor </a>

## Installation 

If you want to run the code on mujoco environments, make sure [mujoco](https://github.com/openai/mujoco-py) is installed on your computer
```bash
git clone https://github.com/cyrille-kone/deep-rl
cd deep-rl
python3 -m venv envsac
source envsac/bin/activate
pip install --upgrade -r requirements.txt
```

## Run 
```bash
python train.py --config "configs/sac.yaml"
```

## Plot results
After training, you can plot results by running
```bash
python plot.py to_plot.log
```

## Original SAC Paper 
https://arxiv.org/abs/1801.01290
