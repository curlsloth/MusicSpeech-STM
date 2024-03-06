#!/bin/bash

#SBATCH --job-name=TAT-Vol2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --output=HPC/slurm/TAT-Vol2_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=ARRAY_TASKS
#SBATCH --mail-type=END


module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('results/corpMetaData/TAT-Vol2.mat', 'TAT-Vol2', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=1-1309 sbatch_script/slurm_STM01_TAT-Vol2.s
