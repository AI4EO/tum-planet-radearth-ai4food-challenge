#!/bin/bash

#SBATCH --output=slurm/%j.out                              
#SBATCH --error=slurm/%j.out                                 
#SBATCH --time=12:00:00
#SBATCH --account=scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source venv/bin/activate
srun "$@"