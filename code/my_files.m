function [myfilelist, curfiles] = my_files(type, corpus, varargin)
%% Read me
% 01/2024 YL
% This function takes in corpus: 
% *** speech ***
% LibriVox
% LibriSpeech
% TIMIT
% EUROM
% SpeechClarity
% TAT-Vol2
% Buckeye
% TTS_Javanese
% *** music ***
% IRMAS
% GarlandEncyclopedia
% Albouy2020Science
% CD
% Type: 'speech' 'music'
% 
% This function looks into the corpus of a certain type
% goes through all the audio recordings with specific extensions
% and collects all necessary file info for later STM analysis
% 
% This function takes in:
%   type: string, corpus type: 'speech' or 'music'
%   corpus: string, corpus name (see the beginning of README)
% 
% This function outputs:
%   myfilelist: cell, containing path, folder name, and filename of the
%           audio recordings in the current corpus
%   curfiles: data structure, containing all file info necessary for STM
%           analysis (will be passed to my_stmAna.m)
% 
%%
% from audio file (waveform) to MS (modulation spectrum)
    P = inputParser;
    P.addRequired('type'); % 'speech', 'music'
    P.addRequired('corpus');
    P.addOptional('debug', 0);
%     P.addOptional('axisOff',0);
    P.addOptional('winLength',4); % window length
    parse(P, type, corpus, varargin{:});    
    
    debug = P.Results.debug;
    wlen = P.Results.winLength;
    wl = num2str(wlen);    
    
    % curfiles: data structure, stores basic info for each audio file in a corpus
    % This data structure will be passed into the my_stmAna.m and provide info for STM analysis
    curfiles.filepath = {}; % path to access each audio file in the corpus
    curfiles.tempMatName = {}; % path to save the STM matrix for each audio file
    curfiles.filename = {}; % file name of each audio file in this corpus
    curfiles.TotalLeng = {}; % total length (sec) of each audio file
    curfiles.TotalSamples = {}; % total sample (sample) of each audio file
    curfiles.fs = {}; % sampling rate of each audio file
    curfiles.langOrinstru = {}; % language or instrumental component of the audio file
    curfiles.VoiceOrNot = {}; % whether the audio file contains voice or not
    curfiles.type = type; % type of the current corpus ('speech' or 'music'), should be the same for all audio files in one corpus
    curfiles.curcorpus = corpus; % corpus name, should be the same for all audio files in one corpus
%% set paths and parameters
    savepath = ['STM_output/MATs/' type '_mat_wl' wl '_' corpus]; % save all the created MAT files
    if ~isfolder(savepath)
        mkdir(savepath);
    end
    curfiles.savepath = savepath;
    
    paramspath = ['STM_output/Survey/' type '_params_' corpus]; % save path for survey result
    if ~isfolder(paramspath)
        mkdir(paramspath);
    end
    curfiles.paramspath = paramspath;

    tempP = genpath(['data/' type 'Corp/' corpus]);
    tempPP = strsplit(tempP,':');
    
    % find out all the audio files in all subfolders
    dirAll = [];
    for pNames = 1:length(tempPP)
        dirAll = vertcat(dirAll,dir(tempPP{pNames}));
    end
    dirAll = struct2table(dirAll);

    % survey all files 
    n = height(dirAll);

    for i = 1:n
        dirAll.folder{i} = erase(dirAll.folder{i},[pwd,'/']); % make it as relative path
    end

    % filter the list of files based on extensions
    switch corpus
        case 'EUROM'
            corp={'N'; 'S'; 'P'}; % 1st letter of the extension, use only recordings of numbers, sentences and passages
            nationality={'D';'E';'F';'G';'H';'I';'N';'S'};% 2nd letter of the extension, nationality of the speakers
            langs = {'Danish'; 'English'; 'French'; 'German'; 'Dutch'; 'Italian'; 'Norwegian'; 'Swedish'};% make sure in the right order as the nationality
            recordingtype={'S'};% 3rd letter of the extension, only use the speech waveform files
            extensions = {};
            for iC = 1:length(corp)
                for iN = 1:length(nationality)
                    for iR = 1:length(recordingtype)
                        extensions{end+1} = ['.' corp{iC} nationality{iN} recordingtype{iR}];% all the possible extensions
                    end
                end
            end
        otherwise
            extensions = {'.mp3', '.WAV', '.wav', '.m4a', '.flac'};% possible extensions of audio recordings
    end
%% get file info
    myfilelist = {}; % just for a quick sanity check (whether the audio file and folder matches reasonably)
    tStart = tic;
    for i = 1:n
        [~,~,ext] = fileparts(dirAll.name(i));
        if contains(ext,extensions)
            tempFileName = [dirAll.folder{i},'/',dirAll.name{i}];

            switch corpus
                case 'TIMIT' %TIMIT multiple audio files have the same name, so use the folderName_audioName
                    folder_pos = find(dirAll.folder{i}=='/', 1, 'last');
                    tempMatName = [savepath '/' dirAll.folder{i}(folder_pos+1:end) '_' dirAll.name{i}(1:end-4)];
                otherwise
                    tempMatName = [savepath '/' dirAll.name{i}(1:end-4)]; % part of the mat file name
            end
%             curfiles.tempMatName{end+1} = tempMatName;

%             if isempty(dir([tempMatName '*_MS2024.mat'])) && ~isempty(tempMatName) % skip the files that have been analyzed & skip the files with empty tempMatName
                if debug 
                    disp(i) 
                end % debug
                myfilelist{end+1,1} = tempFileName; % 1st col of myfilelist: path to the current audio file
                thisfolderpath = dirAll.folder{i}; 
                myfilelist{end,2} = extractAfter(thisfolderpath, 'Corp'); % 2nd col of myfilelist: folder name of the current audio file, e.g.: /LibriVox/eng
                myfilelist{end,3} = dirAll.name{i}; % 3rd col of myfilelist: name of the current audio file

                curfiles.filename{end+1} = dirAll.name{i}(1:end-4);
                curfiles.tempMatName{end+1} = tempMatName;

                tic; % start timing for the main part of processing

                % Modify here to cope with more corpora
                switch corpus
                    case 'TIMIT'
                        lang = 'eng';
                    case 'LibriVox'
                        lang = dirAll.folder{i}(end-2:end);% get the language
                    case 'LibriSpeech'
                        lang = 'eng';
                    case 'Buckeye'
                        lang = 'eng';
                    case 'SpeechClarity'
                        lang = 'engBritish';
                    case 'TAT-Vol2'
                        lang = 'taiwanese';
                    case 'TTS_Javanese'
                        lang = 'javanese';
                    case 'thchs30'
                        lang = 'mandarin';
                    case 'EUROM'
                        ind_lang = find(contains(nationality, dirAll.name{i}(end-1)));% use the matched position to indicate the language
                        lang = langs{ind_lang};
                    case 'BibleTTS/akuapem-twi'
                        lang = 'akuapem';
                    case 'BibleTTS/asante-twi'
                        lang = 'asante';
                    case 'BibleTTS/ewe'
                        lang = 'ewe';
                    case 'BibleTTS/hausa'
                        lang = 'hausa';
                    case 'BibleTTS/lingala'
                        lang = 'lingala';
                    case 'BibleTTS/yoruba'
                        lang = 'yoruba';
                    case 'MediaSpeech/AR'
                        lang = 'arabic';
                    case 'MediaSpeech/ES'
                        lang = 'spanish';
                    case 'MediaSpeech/FR'
                        lang = 'franch';
                    case 'MediaSpeech/TR'
                        lang = 'turkish';
                end

                switch corpus
                    case 'EUROM'
                        % load the EUROM audio file
                        fid=fopen(tempFileName,'r');
                        signal=fread(fid,'int16','l');
                        fclose(fid);

                        fs = 20000;% from EUROM(eurospeech95eurom.pdf): fs = 20khz
                        TotalLeng = length(signal)/fs; % get the total length of the raw signal (in sec)
                        TotalSamples = length(signal); % get the total length of the raw signal (in audio samples)
                    otherwise
                        infoRaw = audioinfo(tempFileName); % get the basic info without loading the whole file
                        fs = infoRaw.SampleRate; % get the original sampling rate
                        TotalLeng = infoRaw.Duration; % get the total length of the raw signal (in sec)
                        TotalSamples = infoRaw.TotalSamples; % get the total length of the raw signal (in audio samples)
                end
                curfiles.TotalLeng{end+1} = TotalLeng;
                curfiles.TotalSamples{end+1} = TotalSamples;
                curfiles.fs{end+1} = fs;
                
                curfiles.filepath{end+1} = tempFileName;
                switch type
                    case 'speech'
                        curfiles.langOrinstru{end+1} = lang;
                        curfiles.VoiceOrNot{end+1} = true; % containing voice
                    case 'music'
                        switch corpus
                            case 'IRMAS'
                                txtLabels = contains(dirAll.name, [dirAll.name{i}(1:end-4),'.txt']); % search for txt (labels) for the current recording
                                if sum(txtLabels) == 1 % should exists and ONLY exists 1 txt for each recording
                                    indLabels = find(txtLabels); % if true, get the index in the dirAll table
                                    tempTxt = textread([dirAll.folder{indLabels},'/',dirAll.name{indLabels}],'%s'); % load and collect the labels
                                else % if none or more than 1 txt exist
                                    tempTxt = string('Error with IRMAS labels!! Check data!'); % record the potential error
                                end
                                curfiles.langOrinstru{end+1} = strjoin(tempTxt,',');  % the instruments involved
                                curfiles.VoiceOrNot{end+1} = sum(contains(tempTxt,'voi')) > 0; % if the recording has human voice
                            otherwise
                                curfiles.langOrinstru{end+1} = 'nonIRMAS';
                                curfiles.VoiceOrNot{end+1} = NaN; % need hand labeling for voice existence
                        end
                end             
%             end
        end
    end
    tEnd = toc(tStart); % total time
    disp(['Total loop time: ', num2str(tEnd, '%.3f'), ' sec']);

end
