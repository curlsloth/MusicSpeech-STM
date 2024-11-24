#!/bin/bash

#SBATCH --job-name=STM14
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=15GB
#SBATCH --time=6:00:00
#SBATCH --output=HPC_slurm/STM14/STM14_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load libsndfile/intel/1.0.31

~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM14_extract_melspectrogram.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-106 HPC_sbatch/STM14/slurm_STM14.s
