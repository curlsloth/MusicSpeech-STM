#!/bin/bash

#SBATCH --job-name=Matlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:05:00
#SBATCH --output=HPC/slurm/output_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=FAIL


module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('results/corpMetaData/TAT-Vol2.mat', 'TAT-Vol2', $SLURM_ARRAY_TASK_ID); exit;"
