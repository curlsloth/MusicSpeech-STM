function [] = STM01_runSTM_HPC(curfiles_path, thisCorpus, array_in)

    tic
    % run with MATLAB R2022b on NYU HPC 
    addpath(genpath('code/'));% dir for customized script
    slurm_output_path = ['HPC_slurm/',strrep(thisCorpus,'/','-')];

    % remove the slurm output folder to make it clean
    if isfolder(slurm_output_path)
        rmdir(slurm_output_path);
    end
    mkdir(slurm_output_path); % recreate a slurm output folder to make it clean

    load(curfiles_path,'curfiles');
    loop_end = min((array_in*100+100), length(curfiles.filename)); % 100 more files or till the end of files
    for fileID = (array_in*100+1):loop_end
        [~, ~, ~, ~, ~] = my_stmAna(curfiles, thisCorpus, fileID);
    end
    disp('Successful STM!')
    toc

end