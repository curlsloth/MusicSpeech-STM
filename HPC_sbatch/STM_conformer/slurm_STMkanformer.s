#!/bin/bash

#SBATCH --job-name=STMkanformer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=HPC_slurm/STM_conformer/STMkanformer-gpu_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge

# singularity exec $(for sqf in /scratch/ac8888/vast/sqfs/*.sqf; do echo "--overlay ${sqf}"; done) /scratch/work/public/singularity/ubuntu-24.04.3.sif /bin/bash

STM_singularity/run-MusicSpeech-STMhpc_conformerGPU.bash python STM_ASM_model.py 0
# Run this: sbatch HPC_sbatch/STM_conformer/slurm_STMkanformer.s