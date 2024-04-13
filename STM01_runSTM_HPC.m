function [] = STM01_runSTM_HPC(curfiles_path, thisCorpus, array_in)

    tic
    % run with MATLAB R2023b on NYU HPC 
    addpath(genpath('MATLAB_functions/'));% dir for customized script

    load(curfiles_path,'curfiles');
    loop_end = min((array_in*100+100), length(curfiles.filename)); % 100 more files or till the end of files
    for fileID = (array_in*100+1):loop_end
        [~, ~, ~, ~, ~] = my_stmAna(curfiles, thisCorpus, fileID);
    end
    disp('Successful STM!')
    toc

end

% CD audio files cannot be run on HPC. Not sure why