#!/bin/bash

#SBATCH --job-name=STM_preViM
#SBATCH --account=torch_pr_578_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STM_branchAuM_preViM-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# Enable PyTorch memory optimization to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STM_singularity/run-MusicSpeech-STMhpc_mamba_torch.bash python3 STM_branchAuM_preViM.py 0 --pretrained_path Vim/vim_s_midclstok_ft_81p6acc.pth
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_STM_branchAuM_preViM.s