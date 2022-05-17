#!/bin/bash

#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00


module unload python
module load anaconda/3 cuda/11.2 cuda/11.1/cudnn/8.1 cuda/11.1/nccl/2.10 mujoco-py
conda activate jax

cd purepg
#gdb -ex r -ex bt --args python main_sync.py
python main_sync.py