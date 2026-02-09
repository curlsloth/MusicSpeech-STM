#!/bin/bash

#SBATCH --job-name=preIN8_ConvNeXt
#SBATCH --account=torch_pr_578_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STM_CoordConvLDAM_preIN8_ConvNeXt-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

STM_singularity/run-MusicSpeech-STMhpc_torchvision.bash python STM_CoordConvLDAM_preIN8_ConvNeXt.py 0
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_STM_CoordConv_preIN8_ConvNeXt.s