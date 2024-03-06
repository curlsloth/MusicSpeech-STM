#!/bin/bash

#SBATCH --job-name=thchs30
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=00:30:00
#SBATCH --output=HPC_slurm/thchs30/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/thchs30.mat','thchs30', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-133 HPC_sbatch/slurm_STM01_thchs30.s
