#!/bin/bash

#SBATCH --job-name=ismir04_genre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:20:00
#SBATCH --output=HPC_slurm/STM01/ismir04_genre/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2023b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/ismir04_genre.mat','ismir04_genre', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-21 HPC_sbatch/STM01/slurm_STM01_ismir04_genre.s
