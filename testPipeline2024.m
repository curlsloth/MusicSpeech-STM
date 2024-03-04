%% Read me
% 01/2024 YL
% sample pipeline of scripts used in analyzing audios to STM
% this script looks into a given corpus
% 1. go through audio recordings one by one and collect basic info
% 2. for each audio recording in the current corpus, cut them into 4s
% excerpts, go through the excerpts, and do STM analysis (waveform -> TF -> STM)
% 
% see more details in the functions
% 
%%
clear all;
clc;
%% parameters
addpath(genpath('code/'));% dir for customized script

% 1. get file info from the current corpus
thisCorpus = 'Buckeye'; 
debug = 0; 

[myfilelist, curfiles] = my_files('speech', thisCorpus); % corpus level info
if size(myfilelist,1) ~= length(curfiles.filename)
    error('File counts not matching!');
end

%% 2. go through each file in the corpus
if debug
    Nfiles = 1; % test
else
    Nfiles = size(myfilelist,1);
end

parfor fileID = 1:Nfiles
% parfor fileID = 1:100


    [indMS, indMS_all, Params, x_axis, y_axis] = my_stmAna(curfiles, thisCorpus, fileID); % audio file level

% 3. if debug, generate a quick overview of the averaged normalized STM of the current corpus
    if debug 
        sqrt_freq_x = sqrt(abs(x_axis));
        sqrt_freq_mat_x = repmat(sqrt_freq_x,length(y_axis),1,size(indMS_all,3));
        indMS_sqrtFreq_x = indMS_all.*sqrt_freq_mat_x;

        sqrt_freq_y = sqrt(abs(y_axis));
        sqrt_freq_mat_y = repmat(sqrt_freq_y',1,length(x_axis),size(indMS_all,3));
        indMS_sqrtFreq = indMS_sqrtFreq_x.*sqrt_freq_mat_y;

        max_indMS = max(max(indMS_sqrtFreq,[],1),[],2);
        max_indMS_mat = repmat(max_indMS,length(y_axis),length(x_axis));
        indMS_norm = indMS_sqrtFreq./max_indMS_mat;

        stm_mat = mean(indMS_norm,3);
    
        inds_x = find(abs(x_axis)<15);
        inds_y = find(abs(y_axis)<3.18 & y_axis>=0);
        stm_mat_plot = stm_mat(inds_y, inds_x);
        my_stm_plot(stm_mat_plot, x_axis(inds_x), y_axis(inds_y), 'axisOff', 1);
    end
end
