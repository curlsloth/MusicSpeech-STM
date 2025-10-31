[Oct 29, 2025]

Use the following command to mount `data`, `model` and `STM_output` folders (also other folders under `vast/ac8888/`) under Singularity

```
singularity exec $(for sqf in /scratch/ac8888/vast/sqfs/*.sqf; do echo "--overlay ${sqf}"; done) /scratch/work/public/singularity/ubuntu-24.04.3.sif /bin/bash
```

The files will be under path `/vast-ac8888/MusicSpeech-STM/`
The script will need to be updated accordingly (not yet)
