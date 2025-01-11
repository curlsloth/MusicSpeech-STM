#!/bin/bash

#SBATCH --job-name=STM11-YAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:0
#SBATCH --mem=50GB
#SBATCH --time=8:00:00
#SBATCH --output=HPC_slurm/STM11/STM11-YAM_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc_GPU.bash python STM11gpu_MLP_YAMNet_corpus.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-7 HPC_sbatch/STM11/slurm_STM11gpu_YAMNet.s
# don't request for too many CPUs as it will use too much memory