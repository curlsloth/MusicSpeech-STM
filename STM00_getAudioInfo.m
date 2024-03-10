% list corpra

clear; clc

speechCorpus = {
    'BibleTTS/akuapem-twi';
    'BibleTTS/asante-twi'
    'BibleTTS/ewe'
    'BibleTTS/hausa'
    'BibleTTS/lingala'
    'BibleTTS/yoruba'
    'Buckeye';
    'EUROM';
    'LibriSpeech';
    'LibriVox';
    'MediaSpeech/AR';
    'MediaSpeech/ES';
    'MediaSpeech/FR';
    'MediaSpeech/TR';
    'MozillaCommonVoice/ab';
    'MozillaCommonVoice/ar';
    'MozillaCommonVoice/ba';
    'MozillaCommonVoice/be';
    'MozillaCommonVoice/bg';
    'MozillaCommonVoice/bn';
    'MozillaCommonVoice/br';
    'MozillaCommonVoice/ca';
    'MozillaCommonVoice/ckb';
    'MozillaCommonVoice/cnh';
    'MozillaCommonVoice/cs';
    'MozillaCommonVoice/cv';
    'MozillaCommonVoice/cy';
    'MozillaCommonVoice/da';
    'MozillaCommonVoice/de';
    'MozillaCommonVoice/dv';
    'MozillaCommonVoice/el';
    'MozillaCommonVoice/en';
    'MozillaCommonVoice/eo';
    'MozillaCommonVoice/es';
    'MozillaCommonVoice/et';
    'MozillaCommonVoice/eu';
    'MozillaCommonVoice/fa';
    'MozillaCommonVoice/fi';
    'MozillaCommonVoice/fr';
    'MozillaCommonVoice/fy-NL';
    'MozillaCommonVoice/ga-IE';
    'MozillaCommonVoice/gl';
    'MozillaCommonVoice/gn';
    'MozillaCommonVoice/hi';
    'MozillaCommonVoice/hu';
    'MozillaCommonVoice/hy-AM';
    'MozillaCommonVoice/id';
    'MozillaCommonVoice/ig';
    'MozillaCommonVoice/it';
    'MozillaCommonVoice/ja';
    'MozillaCommonVoice/ka';
    'MozillaCommonVoice/kab';
    'MozillaCommonVoice/kk';
    'MozillaCommonVoice/kmr';
    'MozillaCommonVoice/ky';
    'MozillaCommonVoice/lg';
    'MozillaCommonVoice/lt';
    'MozillaCommonVoice/ltg';
    'MozillaCommonVoice/lv';
    'MozillaCommonVoice/mhr';
    'MozillaCommonVoice/ml';
    'MozillaCommonVoice/mn';
    'MozillaCommonVoice/mt';
    'MozillaCommonVoice/nan-tw';
    'MozillaCommonVoice/nl';
    'MozillaCommonVoice/oc';
    'MozillaCommonVoice/or';
    'MozillaCommonVoice/pl';
    'MozillaCommonVoice/pt';
    'MozillaCommonVoice/ro';
    'MozillaCommonVoice/ru';
    'MozillaCommonVoice/rw';
    'MozillaCommonVoice/sr';
    'MozillaCommonVoice/sv-SE';
    'MozillaCommonVoice/sw';
    'MozillaCommonVoice/ta';
    'MozillaCommonVoice/th';
    'MozillaCommonVoice/tr';
    'MozillaCommonVoice/tt';
    'MozillaCommonVoice/ug';
    'MozillaCommonVoice/uk';
    'MozillaCommonVoice/ur';
    'MozillaCommonVoice/uz';
    'MozillaCommonVoice/vi';
    'MozillaCommonVoice/yo';
    'MozillaCommonVoice/yue';
    'MozillaCommonVoice/zh-CN';
    'MozillaCommonVoice/zh-TW';
    'primewords_chinese';
    'room_reader';
    'SpeechClarity';
    'TAT-Vol2';
    'thchs30';
    'TIMIT';
    'TTS_Javanese';
    'zeroth_korean';
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

    save_filename = ['STM_output/corpMetaData/',strrep(total_table.name{n},'/','-')]; % if there's a subfolder, replace / by -
    
    % only run the analyses if it hasn't been done
    if ~isfile([save_filename,'.mat']) 
        [~, curfiles] = my_files(total_table.type{n}, total_table.name{n}); % corpus level info
        save(save_filename,'curfiles')
    else
        load(save_filename)
    end

    % generate sbatch script
    sbatch_name = ['slurm_STM01_',strrep(total_table.name{n},'/','-'),'.s'];
    slurm_output_path = ['HPC_slurm/',strrep(total_table.name{n},'/','-')];

    if ~isfolder(slurm_output_path) 
        mkdir(slurm_output_path); % create a slurm output folder to make it clean
        disp(['make folder: ', slurm_output_path])
    end

    array_size = num2str(floor((length(curfiles.filename)-1)/100));
    if array_size > 2000
        disp(total_table.name{n}, ': Array size too big!!')
    end
    sbatch_lines = string();
    sbatch_lines(1) = "#!/bin/bash";
    sbatch_lines(end+1) = "";
    sbatch_lines(end+1) = ['#SBATCH --job-name=',strrep(total_table.name{n},'/','-')];
    sbatch_lines(end+1) = "#SBATCH --nodes=1";
    sbatch_lines(end+1) = "#SBATCH --cpus-per-task=4";
    sbatch_lines(end+1) = "#SBATCH --mem=8GB";
    sbatch_lines(end+1) = "#SBATCH --time=00:40:00";
    sbatch_lines(end+1) = ['#SBATCH --output=',slurm_output_path,'/slurm_%A_%a.out'];
    sbatch_lines(end+1) = "#SBATCH --mail-user=ac8888@nyu.edu";
    sbatch_lines(end+1) = "#SBATCH --mail-type=END";
    sbatch_lines(end+1) = "";
    sbatch_lines(end+1) = "module purge";
    sbatch_lines(end+1) = "module load matlab/2023b";
    sbatch_lines(end+1) = "module load libsndfile/intel/1.0.31";
    sbatch_lines(end+1) = "";
    sbatch_lines(end+1) = "# MATLAB command with input arguments";
    sbatch_lines(end+1) = ['matlab -nodisplay -r "STM01_runSTM_HPC(''',save_filename,'.mat'',''', total_table.name{n},''', $SLURM_ARRAY_TASK_ID); exit;"'];
    sbatch_lines(end+1) = ['# Run this: sbatch --array=0-',num2str(floor((length(curfiles.filename)-1)/100)),' HPC_sbatch/',sbatch_name];
    writelines(sbatch_lines',['HPC_sbatch/',sbatch_name])
end