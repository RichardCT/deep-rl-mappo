### READ THIS ###
    
Run 

conda env create -p [SomePath]\.conda -f environment/environment.yaml

then conda activate [SomePath]\.conda

replacing [SomePath]

It should install everything needed. This assumes Cuda12.4. Change as needed - or remove pytorch gpu altogether if needed.  

Tensorboard is installed as well to monitor training.

to run:
tensorboard --logdir [SomePath]\runs

### Install Ollama on cluster ###

Install Ollama:
`curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz`

### Current Status ###

PPO Algo taken from https://github.com/Chris-hughes10/simple-ppo, with a tutorial explaining everything very nicely by the same guy here https://medium.com/@chris.p.hughes10/understanding-ppo-a-game-changer-in-ai-decision-making-explained-for-rl-newcomers-913a0bc98d2b

Made lots of changes to adapt MAPPO implementation in FrozenLake environment.