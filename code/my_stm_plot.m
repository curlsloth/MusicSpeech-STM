function my_stm_plot(stm_mat, x_axis, y_axis, varargin)
% plot the STM
    P = inputParser;
    P.addRequired('stm_mat');
    P.addRequired('x_axis');
    P.addRequired('y_axis');
    P.addOptional('axisOff',0);
    parse(P, stm_mat, x_axis, y_axis, varargin{:});    
    
    axisOff = P.Results.axisOff;
%% 1 & 2 quadrants
    figure('Name', 'STM');
    
    visData = stm_mat;
%     maxColor = max(max(visData));
    pcolor(x_axis,y_axis,visData); % use the average of 1st and 2nd quadrants
    shading interp;
    colormap(jet);
    cx=caxis;
%     caxis([-maxColor,maxColor]);
    if axisOff
        axis off
    end
    set(gca,'fontsize',12);
%     set(gca,'xscale','log'); % use log on the x-axis to enlarge details in low temporal modulation frequencies
%     xticks([0.5 1 2 3 4 5 6 7 8 9 10]);
    colorbar;
    hold on;
%% 1st quadrant (average of the 1st and 2nd)
    figure('Name', 'STM_1st');
    
    xlen = size(stm_mat,2);
    xq_len = (xlen-1)/2; % length of a quadrant
    xq2 = stm_mat(:,1:xq_len); % 2nd quadrant
    xq1 = stm_mat(:, end-xq_len+1:end); % 1st quadrant
    visData = [stm_mat(:,xq_len+1) (xq1+flip(xq2,2))/2];
    pcolor(x_axis(end-xq_len:end),y_axis,visData); % use the average of 1st and 2nd quadrants
    shading interp;
    colormap(jet);
    cx=caxis;
    if axisOff
        axis off
    end
    set(gca,'fontsize',12);
    set(gca,'xscale','log'); % use log on the x-axis to enlarge details in low temporal modulation frequencies
    xticks([0.5 1 2 3 4 5 6 7 8 9 10]);
    colorbar;
    hold on;
end