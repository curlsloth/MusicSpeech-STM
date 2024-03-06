% list corpra

speechCorpus = {
    'BibleTTS';
    'Buckeye';
    'EUROM';
    'LibriSpeech';
    'LibriVox';
    'MediaSpeech/AR';
    'MediaSpeech/ES';
    'MediaSpeech/FR';
    'MediaSpeech/TR';
    'SpeechClarity';
    'TAT-Vol2';
    'thchs30';
    'TIMIT';
    'TTS_Javanese';
    };

musicCorpus = {
    'Albouy2020Science';
    'CD';
    'GarlandEncyclopedia';
    };

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