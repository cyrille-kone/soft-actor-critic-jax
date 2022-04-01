# Soft Actor-Critic
This is the repo for the project of implementing the algorithm from the paper <a href="https://arxiv.org/pdf/1801.01290.pdf"> Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor </a>

## Installation 

```bash
git clone https://github.com/cyrille-kone/deep-rl
cd deep-rl
python3 -m venv envsac
source envsac/bin/activate
pip install --upgrade -r requirements.txt
```

## Run 
```bash
python train.py --config "configs/example.yaml"
```

# TODO
You may decide to run a part of our experiments by executing the command for the sections below. 

<code> #TODO </code>
## Resources 
# TODO 
https://arxiv.org/abs/1801.01290
# TODO



## Troubleshooting

The mujuco environments may lead to some issues (for Ubuntu at least) :

* <code>fatal error: GL/osmesa.h: No such file or directory</code>

install : <code>sudo apt-get install libosmesa6-dev</code>

* <code>FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'</code>

install : <code>sudo apt-get install patchelf</code>
