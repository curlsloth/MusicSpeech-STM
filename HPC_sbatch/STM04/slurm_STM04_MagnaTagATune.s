#!/bin/bash

#SBATCH --job-name=de_MagnaTagATune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:20:00
#SBATCH --output=HPC_slurm/STM04/MagnaTagATune/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# MATLAB command with input arguments
~/demucs-env/run-demucs.bash python STM04_music_voice_detection.py metaTables metaData_MagnaTagATune.csv $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-258 HPC_sbatch/STM04/slurm_STM04_MagnaTagATune.s
