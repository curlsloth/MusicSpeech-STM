#!/bin/bash

#SBATCH --job-name=STM07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --time=15:00:00
#SBATCH --output=HPC_slurm/STM07_%A.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# MATLAB command with input arguments
~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM07_tSNE_PCA.py
# Run this: sbatch HPC_sbatch/STM07/slurm_STM07.s