function [] = STM01_runSTM_HPC(curfiles_path, thisCorpus, array_in)

    tic
    % run with MATLAB R2022b on NYU HPC 
    addpath(genpath('code/'));% dir for customized script
    load(curfiles_path,'curfiles');
    loop_end = min((array_in*100+100), length(curfiles.filename)); % 100 more files or till the end of files
    parfor fileID = (array_in*100+1):loop_end
        [~, ~, ~, ~, ~] = my_stmAna(curfiles, thisCorpus, fileID);
    end
    disp('Successful STM!')
    toc

end