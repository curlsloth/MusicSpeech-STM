function slurm_reset(slurm_output_path)
    if isfolder(slurm_output_path) 
        delete([slurm_output_path,'/*']); % remove the slurm output folder to make it clean
        disp(['clean: ', slurm_output_path])
    else
        mkdir(slurm_output_path); % recreate a slurm output folder to make it clean
        disp(['make folder: ', slurm_output_path])
    end