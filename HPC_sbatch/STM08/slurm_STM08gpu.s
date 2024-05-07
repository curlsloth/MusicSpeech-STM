#!/bin/bash

#SBATCH --job-name=STM08-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH --gres=gpu:4
#SBATCH --mem=360GB
#SBATCH --output=HPC_slurm/STM08/STM08-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc_GPU.bash python STM08gpu_MLP_corpus.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-1 --time=3-00:00:00 HPC_sbatch/STM08/slurm_STM08gpu.s
