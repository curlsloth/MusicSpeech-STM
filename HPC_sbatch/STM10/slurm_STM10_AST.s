#!/bin/bash

#SBATCH --job-name=STM10_AST
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=2-00:00:00
#SBATCH --output=HPC_slurm/STM10/STM10_AST_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load libsndfile/intel/1.0.31

~/STM_singularity/run-MusicSpeech-STMhpc_transformers.bash python STM10_AST_emb.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0-106 HPC_sbatch/STM10/slurm_STM10_AST.s
