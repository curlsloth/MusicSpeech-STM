#!/bin/bash

#SBATCH --job-name=STMconformer_enhanced2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=4:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STMconformer-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# singularity exec $(for sqf in /scratch/ac8888/vast/sqfs/*.sqf; do echo "--overlay ${sqf}"; done) /scratch/work/public/singularity/ubuntu-24.04.3.sif /bin/bash

STM_singularity/run-MusicSpeech-STMhpc_conformerGPU.bash python STMconformer_enhanced2.py 0 none
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_STMconformer_enhanced2.s