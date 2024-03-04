function [indMS, indMS_all, Params, x_axis, y_axis] = my_stmAna(curfiles, curcorpus, fileID, varargin)
%% Read me 
% 01/2024 YL
% This function goes through each audio recording in the current corpus
% slips each recording into 4s segments (excerpts)
% goes through each excerpts
% analyzes the excerpts with less than 1s silence (audio -> TF -> STM)
% concatenate the STMs, and save parameters 
% 
% this function takes in: 
%   curfiles: data structure, containing all necessary file info of the
%   audio recordings in the current corpus
%   curcorpus: string, name of the current corpus  
%   fileID: number, index of the current analyzed audio recording
%   optional inputs:
%                   debug: debug or not, default 0
%                   winLength: length of the segments (excerpts), default 4
%                               (unit: seconds)
%                   silThreshold: power threshold to define silence,
%                               default 0.02
%                   maxexps: maximum excerpts taken from one same audio
%                               recording, default 30
% 
% this function outputs: 
%   indMS: 2d mat, averaged STM of the current audio recording
%   indMS_all: 3d mat, all the STMs of the excerpts taken from the current
%               audio recording (3rd dim shouldn't be larger than the maxexps)
%   Params: data structure, all the parameters used and generated in the
%           analysis for the excerpts of the current audio recording
%   x_axis: x axis of the STM
%   y_axis: y axis of the STM
% 
%%
% from audio file (waveform) to MS (modulation spectrum)
    P = inputParser;
    P.addRequired('curfile'); % structure of current files
    P.addRequired('curcorpus'); % current corpus name
    P.addRequired('fileID'); % index of the file (used to go through all files in the current corpus)
    P.addOptional('debug', 0);
%     P.addOptional('axisOff',0);
    P.addOptional('winLength',4); % window length
    P.addOptional('silThreshold',0.02); % silence threshold
    P.addOptional('maxexps',30); % maximum excerpts per file
    parse(P, curfiles, curcorpus, fileID, varargin{:});    
    
    if ~strcmp(curcorpus, curfiles.curcorpus)
        error('Corpus name not matching!');
    end
    
    wlen = P.Results.winLength;
    wl = num2str(wlen);
    debug = P.Results.debug;
%     axisOff = P.Results.axisOff;      
    silenceThresh = P.Results.silThreshold;
    max_exps = P.Results.maxexps; 
    savepath = curfiles.savepath;
    paramspath = curfiles.paramspath;
    type = curfiles.type;
    
    curFilePath = curfiles.filepath{fileID};
    curFileName = curfiles.filename{fileID};
    TotalLeng = curfiles.TotalLeng{fileID}; 
    fs = curfiles.fs{fileID}; 
    TotalSamples = curfiles.TotalSamples{fileID};
    tempMatName = curfiles.tempMatName{fileID}; % to save the averaged MS (indMS)
    disp(curFilePath) % display the file path in the command window
%% main analysis: audio - TF - MS
    tStart = tic;
    
    indMS = [];
    indMS_all = [];
    Params = [];
    x_axis = [];
    y_axis = [];
    if TotalLeng >= wlen
        startPoint0 = 1;
        if strcmp(curcorpus, 'LibriVox')
            startPoint0 = 1+fs*30; % for Librivox, excluding the first 30s of the copyright statement
        end
        n_exps = 1;
        startPoint = startPoint0;
        sil_L_a = [];% lengths of the silent gaps of analyzed excerpts of the current file
        
        % process each audio recording by excerpts of 4s
        while (startPoint + wlen*fs - 1 <= TotalSamples && n_exps <= max_exps)
%             startPoint % debug
%             n_exps % debug
            endPoint = startPoint + wlen*fs -1; % endPoint of the current excerpt
            sample = [startPoint, endPoint];
        
            % load the excerpt for analysis
            if strcmp(curcorpus, 'EUROM')
                % load the EUROM audio file
                fid=fopen(curFilePath,'r');
                signal=fread(fid,'int16','l');
                fclose(fid);
                sig_excpt = signal(startPoint:endPoint); % load the current 4s excerpt
            else
                [sig_excpt,fs]=audioread(curFilePath, sample); % load the current 4s excerpt
            end     
            
            % STM analysis (only excerpts having silence < 1s will have non-empty outputSTM)
            outputSTM = stm_audio(sig_excpt, silenceThresh, fs);
            if ~isempty(outputSTM)
                MS_excpt = outputSTM.MS;
                indMS(:,:,n_exps) = MS_excpt.orig_MS; % get all the MS info
                x_axis = MS_excpt.x_axis;
                y_axis = MS_excpt.y_axis;
                sil_L = outputSTM.sil_L;
                sil_L_a = [sil_L_a, max(sil_L)/fs];% get all the lengths of silent gaps in the analyzed excepts
                n_exps = n_exps + 1;
                clear MS sig_excpt sil_L;
            end
            startPoint = startPoint + wlen*fs; % start point of the next excerpt, still using the original fs to do the separation
        end
        
%% collect survey parameters (Params structure)
        if ~isempty(indMS)
            indMS_all = indMS; %all MS for this corpus (3D mat)
            indMS = mean(indMS, 3); %averaged MS for this corpus (2D mat)

            Params.filepath = curFilePath;
            Params.filename = curFileName;
            Params.LangOrInstru = curfiles.langOrinstru{fileID};
            switch type
                case 'speech'
                    Params.VoiOrNot = true; % containing voice
                case 'music'
                    switch curcorpus
                        case 'IRMAS'
                            Params.VoiOrNot = sum(contains(curfiles.langOrinstru{fileID},'voi')) > 0; % if the recording has human voice
                        otherwise
                            Params.VoiOrNot = NaN; % need hand labeling for voice existence
                    end
            end
            Params.startPoint = startPoint0;
            Params.endPoint = endPoint; % end point of the current excerpt            
            Params.FSkhz = fs/1000; % sampling rate (in kHz)
            Params.expdur = (endPoint-startPoint0+1)/fs; % the length of total analyzed data
            Params.maxsil = max(sil_L_a); % the length of the longest silence of the analyzed excerpts (in sec), should be smaller than 1
            Params.totalLengCur = wlen*(n_exps-1); % total length of the signal used for analysis (in sec, should not exceed 120s)
            Params.x_axis = x_axis;
            Params.y_axis = y_axis;

            save([tempMatName '_MS2024.mat'], 'indMS'); % save the averaged MAT file of the MS results
            save([paramspath '/' curFileName '_Params.mat'], 'Params');
        end
    end
    tEnd = toc(tStart); % total time
    sprintf('Total loop time: %.3f sec',tEnd)
%% save survey results to folder
%     save([savepath '/x_' type '_' wl '_' curFileName '.mat'], 'x_axis'); 
%     save([savepath '/y_' type '_' wl '_' curFileName '.mat'], 'y_axis');
%     Params = cell2table(Params,'VariableNames',{'filename','langOrInstru', 'VoiceOrNot','startPoint','endPoint','SamplingRate','analyzedLength','silenceLength', 'usedLength'});
%     writetable(Params, [svypath '/' type 'Survey2024_' wl '_' curfile '.xlsx'], 'WriteMode','Append');
    
    
%% generic STM analysis for 1 audio excerpt 
function outputSTM = stm_audio(audio_excpt, silenceThresh, fs)
        audio_excpt = mean(audio_excpt,2); % considering the recordings which have more than 1 channels, take the mean value
        audio_excpt = audio_excpt./(rms(audio_excpt)*10); % RMS normalization, equalize the amplitude (loudness perception) to rms = 0.1

        % check silence period, only analyze excerpts with less than 1s of silence
        sig_belowThresh = bwlabel(abs(audio_excpt)<silenceThresh); % cluster the adjacent samples with threshold lower than 0.05
        sig_belowThresh(sig_belowThresh==0) = []; % exclude the 0, which represents the amplitude larger than threshold
        sil_Len = groupcounts(sig_belowThresh); % count the length (sample) of each cluster, sil = silence

%             only analyze excerpt with less than 1s silence
        if max(sil_Len)/fs < 1
%             calculate and save the modulation spectrum of this excerpt
            outputSTM.sil_L = sil_Len;
            [p,q]=rat(16000/fs); % resample to 16000 hz
            audio_excpt = resample(audio_excpt, p, q); % resample requires double datatype
            fs_r = 16000; % update the fs, ONLY use for MS calculation, r for resample
            audio_excpt = single(audio_excpt); % convert to single datatype to save memory
            TF = STM_CreateTF(audio_excpt, fs_r, 'gauss'); % STM_CreateTF by default use the 1st channel of stereo
            MS = STM_Filter_Mod(TF, [0 inf], [0 inf]);
            outputSTM.MS = MS;
            outputSTM.x_axis = MS.x_axis;
            outputSTM.y_axis = MS.y_axis;
        else
            outputSTM = [];
        end
end
end
