
#!/bin/bash

#SBATCH --job-name=TTS_Javanese
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --output=HPC_slurm/TTS_Javanese/slurm_%A_%a.out
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2022b
module load libsndfile/intel/1.0.31

# MATLAB command with input arguments
matlab -nodisplay -r "STM01_runSTM_HPC('STM_output/corpMetaData/TTS_Javanese.mat','TTS_Javanese', $SLURM_ARRAY_TASK_ID); exit;"
# Run this: sbatch --array=1-5822 HPC_sbatch/slurm_STM01_TTS_Javanese.s
