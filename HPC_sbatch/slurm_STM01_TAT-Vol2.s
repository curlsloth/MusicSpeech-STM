#!/bin/bash

#SBATCH --job-name=TAT-Vol2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=HPC_slurm/TAT-Vol2/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "slurm_reset('HPC_slurm/TAT-Vol2'); STM01_runSTM_HPC('STM_output/corpMetaData/TAT-Vol2.mat','TAT-Vol2', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-13 HPC_sbatch/slurm_STM01_TAT-Vol2.s
