#!/bin/bash

#SBATCH --job-name=HiltonMoser2022_song
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:20:00
#SBATCH --output=HPC_slurm/STM01/HiltonMoser2022_song/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2023b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/HiltonMoser2022_song.mat','HiltonMoser2022_song', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=0-8 HPC_sbatch/STM01/slurm_STM01_HiltonMoser2022_song.s
