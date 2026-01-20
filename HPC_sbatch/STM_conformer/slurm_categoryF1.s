#!/bin/bash

#SBATCH --job-name=F1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STM-F1_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

STM_singularity/run-MusicSpeech-STMhpc_conformerGPU.bash python categoryF1.py --dir model/STM/ASM_Enhanced_corpora_categories/standard/ckpt/2026-01-18_07-29
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_categoryF1.s