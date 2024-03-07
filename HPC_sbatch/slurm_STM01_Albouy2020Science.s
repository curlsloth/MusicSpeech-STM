#!/bin/bash

#SBATCH --job-name=Albouy2020Science
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=HPC_slurm/Albouy2020Science/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/Albouy2020Science.mat','Albouy2020Science', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-1 HPC_sbatch/slurm_STM01_Albouy2020Science.s
