#!/bin/bash
#SBATCH --partition=M2
#SBATCH --qos=q_a_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --job-name=ppo-llm
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --time=06:00:00

module load cuda/12.2.2
eval "$(conda shell.bash hook)" 
conda activate dl_mappo

python3 main.py