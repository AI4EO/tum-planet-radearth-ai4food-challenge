#!/bin/bash

#SBATCH --output=slurm/%j.out                              
#SBATCH --error=slurm/%j.out                                 
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=cmlgrad02,cml12,cmlgrad05

source venv/bin/activate
srun "$@"