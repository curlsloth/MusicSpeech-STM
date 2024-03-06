% list corpra

speechCorpus = {'MediaSpeech/TR';
    'TAT-Vol2';
    'LibriVox';
    'LibriSpeech';
    'TIMIT';
    'EUROM';
    'SpeechClarity';
    'TAT-Vol2';
    'Buckeye';
    'TTS_Javanese'};

musicCorpus = {'GarlandEncyclopedia';
    'Albouy2020Science';
    'CD'};

% make table
speech_table = table(speechCorpus,'VariableNames',"name");
speech_table.type = repmat({'speech'}, length(speechCorpus), 1);

music_table = table(musicCorpus,'VariableNames',"name");
music_table.type = repmat({'music'}, length(musicCorpus), 1);

total_table = [speech_table;music_table];

% run script in a loop

addpath(genpath('code/'));% dir for customized script

for n = 1:height(total_table)
    [~, curfiles] = my_files(total_table.type{n}, total_table.name{n}); % corpus level info
    save_filename = ['results/corpMetaData/',strrep(total_table.name{n},'/','-')]; % if there's a subfolder, replace / by -
    save(save_filename,'curfiles')
end