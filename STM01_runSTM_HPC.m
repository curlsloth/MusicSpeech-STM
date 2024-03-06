function [] = STM01_runSTM_HPC(curfiles_path, thisCorpus, fileID)

    tic
    % run with MATLAB R2022b on NYU HPC 
    addpath(genpath('code/'));% dir for customized script
    load(curfiles_path,'curfiles');
    [~, ~, ~, ~, ~] = my_stmAna(curfiles, thisCorpus, fileID);
    disp('Successful STM!')
    toc

end