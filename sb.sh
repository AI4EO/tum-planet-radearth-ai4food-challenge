#!/bin/bash

#SBATCH --output=slurm/%j.out                              
#SBATCH --error=slurm/%j.out                                 
#SBATCH --time=12:00:00
#SBATCH --account=scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=cmlgrad02

source venv/bin/activate
srun "$@"