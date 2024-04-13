function [WelchOut] = my_Welch(input, fs, windlength, overlap)
% Use Welch method instead of full-FFT to calculate modulation spectrum: 
% parse the input along the time axis into segments of length = windlength(seconds), window overlap = overlap(%),
% apply Hanning taper to each segment to eliminate edge effect,
% 2D fft for each tapered segment,
% get the averaged fft2 result across segments. 
% 
% eg: Welch using 4s window with window overlap of 50%: 
% my_Welch(input, fs, windlength=4, overlap = 0.5);

nfr = size(input, 1); % total frequency bands of the input data
lengT = size(input, 2); % total length of the input data (Time-Frequency matrix, 2nd dimension = time)
wl = fs * windlength; % window length in unit of data points
rs = 125; % Resampled rate for the spectrogram time axis (arguement from STM_CreateTF.m)
[p,q] = rat(rs/fs); % resample the length of the window to the same time scale as the input (TF)
wl_r = wl*(p/q); % wl_r = resampled window length
overlapl = round(overlap * wl_r); % overlapped length in unit of data points
st = 1; % initialize the start timepoint of window
n = ceil(lengT/(wl_r - overlapl)); % in total n segments

h = hanning(wl_r); % generate the Hanning taper of length of the segment, to make sure the power = 0 at the start and end timepoints
h_mat = repmat(h', nfr, 1); % repeat the Hanning taper for all frequency bands

WelchOut = zeros(nfr, wl_r); % pre-allocate Welch output array

WelchTemp = [];
for i = 1:n
    et = st + wl_r - 1; % end timepoint of window
    if et <= lengT
        seg = input(:, st:et);
        
        seg_t = h_mat .* seg; % tapering the segment of signal
%         seg_t = seg; % test, without Hanning
        
        seg_w = fft2(seg_t); % get the 2D fft result of the current tapered segment
        WelchTemp(:,:,i) = seg_w; % store the 2D fft result
        st = st + wl_r - overlapl; % update the start timepoint of the next segment
        
%         figure;
%         visData = abs(seg_w);
%         maxColor = max(max(visData))
%         pcolor(visData);x=caxis;caxis([-maxColor,maxColor]);
%         shading interp;
%         colormap(jet(128));
    else
        break;
    end
end

WelchOut = mean(WelchTemp,3);

% figure; 
% visData1 = abs(WelchOut);
% maxColor1 = max(max(visData1))
% pcolor(visData1);x=caxis;caxis([-maxColor1,maxColor1]);
% shading interp;
% colormap(jet(128));


