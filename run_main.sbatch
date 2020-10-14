#!/bin/bash

#SBATCH --job-name=ica_components
#SBATCH --output=slurm.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00

SRCDIR=$HOME/proj/texturemodeling/sPyrDerNet
cd $SRCDIR

python main.py > ica_comp.out




