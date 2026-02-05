#!/bin/bash

#SBATCH --job-name=STMasm_mixer_kan
#SBATCH --account=torch_pr_578_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STMasm_mixer_kan-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

STM_singularity/run-MusicSpeech-STMhpc_conformerGPU_torch.bash python3 STMasm_mixer_kan.py 0
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_STMasm_mixer_kan.s