#!/bin/bash

#SBATCH --job-name=STM08
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=300GB
#SBATCH --output=HPC_slurm/STM08/STM08_%A.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM08_MLP_STM_corpus.py
# Run this: sbatch --time=12:00:00 HPC_sbatch/STM08/slurm_STM08.s
