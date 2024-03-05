%% parameters
addpath(genpath('code/'));% dir for customized script

% 1. get file info from the current corpus
thisCorpus = 'MediaSpeech/TR'; 

save_filename = ['results/corpMetaData/',strrep(thisCorpus,'/','-')];

[~, curfiles] = my_files('speech', thisCorpus); % corpus level info

save(save_filename,'curfiles')