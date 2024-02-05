function [myfilelist, curfiles] = my_files(type, corpus, varargin)
%% Read me
% 01/2024 YL
% This function looks into the corpus of a certain type
% goes through all the audio recordings with specific extensions
% and collects all necessary file info for later STM analysis
% 
% This function takes in:
%   type: string, corpus type: 'speech' or 'music'
%   corpus: string, corpus name
% 
% This function outputs:
%   myfilelist: cell, containing path, folder name, and filename of the
%           audio recordings in the current corpus
%   curfiles: data structure, containing all file info necessary for STM
%           analysis
% 
%%
% from audio file (waveform) to MS (modulation spectrum)
    P = inputParser;
    P.addRequired('type'); % 'speech', 'music'
    P.addRequired('corpus');
    P.addOptional('debug', 0);
    P.addOptional('axisOff',0);
    P.addOptional('winLength',4); % window length
    parse(P, type, corpus, varargin{:});    
    
    debug = P.Results.debug;
    wlen = P.Results.winLength;
    wl = num2str(wlen);
    
    curfiles.tempMatName = {};
    curfiles.filename = {};
    curfiles.TotalLeng = {};
    curfiles.TotalSamples = {};
    curfiles.fs = {};
    curfiles.filepath = {};
    curfiles.langOrinstru = {};
    curfiles.VoiceOrNot = {};
    curfiles.type = type;
    curfiles.curcorpus = corpus;
%% set paths and parameters
    savepath = [pwd '/results/MATs/' type '_mat_' wl '_' corpus]; % save all the created MAT files
    if ~isfolder(savepath)
        mkdir(savepath);
    end
    curfiles.savepath = savepath;
    
    paramspath = [pwd '/results/Survey/' type '_params_' corpus]; % save path for survey result
    if ~isfolder(paramspath)
        mkdir(paramspath);
    end
    curfiles.paramspath = paramspath;

    tempP = genpath([pwd,'/data/' type 'Corp/' corpus]);
    tempPP = strsplit(tempP,':');
    
    % find out all the audio files in all subfolders
    dirAll = [];
    for pNames = 1:length(tempPP)
        dirAll = vertcat(dirAll,dir(tempPP{pNames}));
    end
    dirAll = struct2table(dirAll);

    % survey all files 
    n = height(dirAll);
    % n = 8000; % test

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
            extensions = {'.mp3', '.WAV', '.wav', '.m4a'};% possible extensions of audio recordings
    end
%% get file info
    myfilelist = {};
    tStart = tic;
    for i = 1:n
        if contains(dirAll.name(i),extensions)
            tempFileName = [dirAll.folder{i},'/',dirAll.name{i}];
            myfilelist{end+1,1} = tempFileName;
            myfilelist{end,2} = dirAll.folder{i};
            myfilelist{end,3} = dirAll.name{i};
            curfiles.filename{end+1} = dirAll.name{i}(1:end-4);

            switch corpus
                case 'TIMIT' %TIMIT multiple audio files have the same name, so use the folderName_audioName
                    folder_pos = find(dirAll.folder{i}=='/', 1, 'last');
                    tempMatName = [savepath '/' dirAll.folder{i}(folder_pos+1:end) '_' dirAll.name{i}(1:end-4)];
                case 'IRMAS'
                    if contains(dirAll(i).name,'.txt')
                        tempTxt = textread([dirAll(i).folder,'/',dirAll(i).name],'%s');
                        if length(tempTxt)<10 % if it has more than 10 words, then it's probably not correspond to audio file
                             tempWavName = [dirAll(i).folder,'/',dirAll(i).name];
                             tempWavName = [tempWavName(1:end-4),'.wav'];
                             tempMatName = [savepath '/' dirAll(i).name(1:end-4)]; % part of the mat file name
                        end
                    end
                otherwise
                    tempMatName = [savepath '/' dirAll.name{i}(1:end-4)]; % part of the mat file name
            end
            curfiles.tempMatName{end+1} = tempMatName;

            if isempty(dir([tempMatName '*_MS2024.mat'])) % skip the files that have been analyzed
                if debug 
                    i 
                end % debug
                tic; % start timing for the main part of processing

                % Modify here to cope with more corpora
                switch corpus
                    case 'TIMIT'
                        lang = 'eng';
                    case 'LibriVox'
                        lang = dirAll.folder{i}(end-2:end);% get the language
                    case 'UK'
                        lang = 'engUK';
                    case 'EUROM'
                        ind_lang = find(contains(nationality, dirAll.name{i}(end-1)));% use the matched position to indicate the language
                        lang = langs{ind_lang};
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
                                curfiles.langOrinstru{end+1} = strjoin(tempTxt,',');  % the instruments involved
                                curfiles.VoiceOrNot{end+1} = sum(contains(tempTxt,'voi')) > 0; % if the recording has human voice
                            otherwise
                                curfiles.langOrinstru{end+1} = 'nonIRMAS';
                                curfiles.VoiceOrNot{end+1} = NaN; % need hand labeling for voice existence
                        end
                end             
            end
        end
    end
    tEnd = toc(tStart); % total time
    sprintf('Total loop time: %.3f sec',tEnd)

end
