#!/bin/bash

#SBATCH --job-name=de_fma_large
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:20:00
#SBATCH --output=HPC_slurm/STM04/fma_large/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# MATLAB command with input arguments
~/demucs-env/run-demucs.bash python STM04_music_voice_detection.py metaTables metaData_fma_large.csv $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-1062 HPC_sbatch/STM04/slurm_STM04_fma_large.s
