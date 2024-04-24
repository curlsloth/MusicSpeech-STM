#!/bin/bash

#SBATCH --job-name=STM07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --time=10:00:00
#SBATCH --output=HPC_slurm/STM07/STM07_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

~/STM_singularity/run-MusicSpeech-STMhpc.bash python STM07_tSNE_PCA.py $SLURM_ARRAY_TASK_ID
# Run this: sbatch --array=0 HPC_sbatch/STM07/slurm_STM07_80RAM.s
# array_input: perplexity of t-SNE, 0 for PCA. Good for perplexity up to 200