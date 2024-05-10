#!/bin/bash

#SBATCH --job-name=STM09
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00
#SBATCH --output=HPC_slurm/STM09/STM09_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM09_sklearn_classifiers.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=4-6 HPC_sbatch/STM09/slurm_STM09.s
