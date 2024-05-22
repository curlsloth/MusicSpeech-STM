# Conda environments

Most of the scripts were run under the conda environment `MusicSpeech-STMlite.yaml` on a mac laptop, with a few exceptions: 

- The script `STM04_music_voice_detection.py` was run under the conda environment `demucs.yaml`.

- The script `STM08gpu_MLP_STM_corpus.py`, `STM09_sklearn_classifiers.py`, `STM11gpu_MLP_VGGish_corpus.py`, `STM11gpu_MLP_YAMNet_corpus.py` were run under the conda environment `MusicSpeech-STMhpc_GPU.yaml` on a high performance computing cluster. See `./HPC_sbatch` for for information regarding the required computational resources.

The other environments were used for testing purposes and saved as a record.
