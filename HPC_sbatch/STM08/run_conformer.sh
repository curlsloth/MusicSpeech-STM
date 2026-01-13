#!/bin/bash
#SBATCH --job-name=conformer_stm
#SBATCH --output=HPC_slurm/STM08/conformer_%A_%a.out
#SBATCH --error=HPC_slurm/STM08/conformer_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust based on your HPC environment)
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Activate conda environment
source ~/.bashrc
conda activate MusicSpeech-STM

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Change to working directory
cd /vast/ac8888/MusicSpeech-STM

# Run the script
echo "Starting Conformer training with mode $SLURM_ARRAY_TASK_ID"
python STM08gpu_Conformer_STM_corpus.py $SLURM_ARRAY_TASK_ID

echo "Job completed at $(date)"
