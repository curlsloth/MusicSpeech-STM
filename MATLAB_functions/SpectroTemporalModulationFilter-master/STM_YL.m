%% The script adopted Adeen Flinker's STM toolbox, 
% do the FM (spectro-temporal modulation) of the auditory input, 
% plot the original waveform, time-freq spectrum, and the modulation domain of the recording
% Yike Li, 03/29/2022, 
%%
clear;
clc;
%%
% cd '/Users/echo/Documents/GoodGoodStudy/NYU_Psych/StudyHard/Go/Poeppel/MaybeThesis/SpectroTemporalModulationFilter-master/wavs';
cd '/Users/echo/Documents/GoodGoodStudy/NYU_Psych/StudyHard/Go/Poeppel/MaybeThesis/methodcomp/musicCorp';

% files = dir('music*.wav');
files = dir("(02) dont kill the whale-3.wav");
%% calculate all the MS info
n = length(files);
AllMS = [];
for i = 1:n
    i % dubug
%     filename = 'LDC93S1.wav'; % test
    filename = files(i).name;
    [signal, fs] = audioread(filename); % Adeen used readsph func from the Voicebox
    signal = signal(1:fs*4); % get the first 4 sec for analysis
    [p,q]=rat(16000/fs);
    signal = resample(signal, p, q); % resample all recordings to 16000hz
    fs = 16000;
%     signal_padded = zeros(1,75500);
    if size(signal, 2) == 2
        signal_mono = (signal(:,1) + signal(:,2))/2; % convert stereo recordings to mono
%         signal_padded(1:length(signal_mono)) = signal_mono;
    else
        signal_mono = signal;
%         signal_padded(1:length(signal)) = signal;
    end
%     TF = STM_CreateTF(signal_padded, fs, 'gauss'); 
    TF = STM_CreateTF(signal_mono, fs, 'gauss'); % STM_CreateTF by default use the 1st channel of stereo

    MS = STM_Filter_Mod(TF, [0 inf], [0 inf]); % fft2 method
    AllMS(:,:,i) = MS.orig_MS; % get all the MS info
%     AllTF(:,:,i) = TF.TF;
    
%     MS = STM_Welch(TF); % Welch method
%     AllMS(:,:,i) = MS.orig_MS; % get all the MS info

end

AllMS(:,MS.x_axis==0,:) = []; % remove the data at 0 Hz to avoid the issue of dividing 0
freqAxis = MS.x_axis(MS.x_axis~=0);

% normalize the power
sqrt_freq = sqrt(abs(freqAxis));
sqrt_freq_mat = repmat(sqrt_freq,length(MS.y_axis),1,n);
AllMS_sqrtFreq = AllMS./sqrt_freq_mat;

max_AllMS = max(max(AllMS_sqrtFreq,[],1),[],2);
max_AllMS_mat = repmat(max_AllMS,length(MS.y_axis),length(MS.x_axis)-1);
AllMS_norm = AllMS_sqrtFreq./max_AllMS_mat;

% sqrt_freq = sqrt(abs(TF.x_axis));
% sqrt_freq_mat = repmat(sqrt_freq,150,1,16);
% AllTF_sqrtFreq = AllTF./sqrt_freq_mat;
% 
% max_AllTF = max(max(AllTF_sqrtFreq,[],1),[],2);
% max_AllTF_mat = repmat(max_AllTF,150,500);
% AllTF_norm = AllTF_sqrtFreq./max_AllTF_mat;

musicMS = AllMS_norm;
% speechMS = AllMS_norm;
% musicTF = AllTF_norm;

% get the parameter from the last (or whichever) recording for latter group-plotting
inds_x = find(abs(freqAxis)<15);
inds_y = find(abs(MS.y_axis)<3.18 & MS.y_axis>=0);
%% plot the waveform and matched spectrum (time-frequency)
figure(1);
t = (length(signal_mono)-1)/fs;
subplot(211); % ploting the waveform 
plot(0:1/fs:t, signal);
axis off; 
xlim([0, t]);
ylim([-0.2, 0.22]); % adopted from Adeen
set(gca, 'fontsize', 22, 'box', 'off');
set(gcf, 'color', 'w', 'Position', [680 552 602 546]);

subplot(212); % plotting the Time-Frequency Spectrum
pcolor(TF.x_axis,TF.y_axis,TF.TFlog); shading interp;colormap(jet(128));caxis([0 60]);% colorbar
set(gca,'fontsize',22,'Xtick',[0:0.5:max(get(gca,'Xtick'))],'Ytick',[20:40:150]);
set(gca,'YtickLabel',round(TF.Args.CB_CenterFrs(get(gca,'Ytick'))/10)*10);

%% plot the FM (spectro-temporal modulation) graph (modulation domain)
% the heat map showing the weight of each component
figure;

visData = mean(musicMS, 3);
% visData = mean(speechMS, 3);
% visData = mean(musicMS - speechMS,3);

% [h,p,~,stat]=ttest2(musicMS,speechMS,'dim',3, 'Alpha', 0.01);
% visData = abs(stat.tstat);
% p(p>0.01) = 1;
% p_plot = 1-p; % so that the fig will only plot the p lower than 0.01; 
% the lower p-value is, the higher power in the heatmap will be
% visData = p_plot; 

maxColor = max(max(visData));
mn = visData;
pcolor(freqAxis(inds_x),MS.y_axis(inds_y),mn(inds_y,inds_x));
shading interp;
% cx=caxis;caxis([-maxColor,maxColor]);
% colormap(jet(128));
colormap(parula);
set(gca,'ColorScale','log')
set(gca,'fontsize',12);
hold on;
