#!/bin/bash

#SBATCH --job-name=STM15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:0
#SBATCH --mem=100GB
#SBATCH --time=6:00:00
#SBATCH --output=HPC_slurm/STM15/STM15_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc_GPU.bash python STM15gpu_MLP_melspectrogram_corpus.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-7 HPC_sbatch/STM15/slurm_STM15gpu_melspectrogram.s
# don't request for too many CPUs as it will use too much memory