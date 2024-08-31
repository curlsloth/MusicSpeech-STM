#!/bin/bash

#SBATCH --job-name=STM08-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:0
#SBATCH --mem=369GB
#SBATCH --time=12:00:00
#SBATCH --output=HPC_slurm/STM08/STM08-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc_GPU.bash python STM08gpu_MLP_STM_corpus.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-7 HPC_sbatch/STM08/slurm_STM08gpu.s
# don't request for too many CPUs as it will use too much memory