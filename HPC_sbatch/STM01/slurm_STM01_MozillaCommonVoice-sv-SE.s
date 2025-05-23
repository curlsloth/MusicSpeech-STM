#!/bin/bash

#SBATCH --job-name=MozillaCommonVoice-sv-SE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:40:00
#SBATCH --output=HPC_slurm/STM01/MozillaCommonVoice-sv-SE/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2023b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/MozillaCommonVoice-sv-SE.mat','MozillaCommonVoice/sv-SE', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-99 HPC_sbatch/STM01/slurm_STM01_MozillaCommonVoice-sv-SE.s
