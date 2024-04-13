#!/bin/bash

#SBATCH --job-name=Matlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:05:00

module purge
module load matlab/2021a

export MATLAB_PREFDIR=$(mktemp -d $SLURM_JOBTMP/matlab-XXXX)
export MATLAB_LOG_DIR=$SLURM_JOBTMP

# MATLAB command with input arguments
matlab -nodisplay -r "test_script($SLURM_ARRAY_TASK_ID); exit;"
