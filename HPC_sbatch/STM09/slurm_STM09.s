#!/bin/bash

#SBATCH --job-name=STM09
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150GB
#SBATCH --time=18:00:00
#SBATCH --output=HPC_slurm/STM09/STM09_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM08_MLP_corpus.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-3 HPC_sbatch/STM09/slurm_STM09.s
