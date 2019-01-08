function [] = plotData(allNames, allTrialLengths, allTrialSheathsR, allTrialSheathsG, saveDir, cur_dir)

cd(saveDir);
%% Figure 1: Plot histogram (Num Cells)
figure;
for k = 1 : length(allNames)
    color = rand(1, 3);
    histogram(allTrialLengths{k}, 200, 'facecolor', color ,'facealpha',.9,'edgecolor',color, 'DisplayName',cell2mat(allNames(k)))
    hold on;
end
box off
xlabel('Length (um)');
ylabel('Total Count (Num Cells)');
legend('show');
title('Length of Fibers Absolute number');

%print figure
filename = strcat('Length (Num Cells) histogram');
print(filename,'-dpng')
hold off;

%% 2nd figure (Frequency Grouped)
figure;
minL = 1000;
maxL = 0;
allFreq = [];
for k = 1 : length(allNames)
    color = rand(1, 3);
    lines = cell2mat(allTrialLengths(k));
    counts = histcounts(lines);
    freq = 100 * (counts ./ length(lines));
    freq = freq';
    if ~isempty(allFreq)
        [maxLength, ind]= max([length(allFreq(:,1)), length(freq)]);
        if ind == 2
            allFreq = padarray(allFreq, [length(freq) - length(allFreq(:, 1)), 0], 'post');
        else
            freq = padarray(freq, [length(allFreq(:,1)) - length(freq), 0], 'post');
        end
    end
    
    % Set max for new axis
    if maxL < max(lines)
        maxL = max(lines);
    end
    
    % Set min for new axis
    if minL > min(lines)
        minL = min(lines);
    end
    allFreq = [allFreq, freq];
end

%% Make Axis
d = linspace(minL, maxL, length(allFreq(:, 1))); % augment scale that is same size as bins
%bins =  10: 10 : (length(allFreq(:, 1)) + 1) *10 - 1; % axis for the bar below
bar(d, allFreq, 'hist', 'grouped', 'barwidth', 1);
hold on

box off
xlabel('Length (um)');
ylabel('Normalized Count [%]');
legend(allNames);
title('Length of Fibers Frequency Grouped');

%print figure
filename = strcat('Length (Frequency Grouped) histogram');
print(filename,'-dpng')
hold off;

%% 3rd Figure (Frequency Overlay)
figure;
newFreq = allFreq';
for k = 1 : length(allNames)
    color = rand(1, 3);
    bar(d, newFreq(k, :), 'facecolor', color, 'facealpha',.9,'edgecolor',color,'barwidth', 1,'DisplayName',cell2mat(allNames(k)));
    hold on;
end
box off
xlabel('Length (um)');
ylabel('Normalized Count [%]');
legend('show');
title('Length of Fibers Frequency Overlay');

%print figure
filename = strcat('Length (Frequency Overlay) histogram');
print(filename,'-dpng')
hold off;

%% 4th Figure log scale:
figure;
newLength = [];
for k = 1 : length(allNames)
    color = rand(1, 3);
    scaled = allTrialLengths{k};
    logged = log10(scaled); % logarithmic
    
    [counts, binValues] = hist(logged, 10);
    %% Works kind of
    normalizedCounts = 100 * counts / sum(counts);
    %plot(binValues, normalizedCounts, 'o');
    % bar(binValues, normalizedCounts, 'barwidth', 1);
    % xlabel('Input Value');
    % ylabel('Normalized Count [%]');
    % hold on;
    % get an estimate of mu and sigma from the "data"
    [muhat,sigma] = normfit(logged);
    % use muhat and sigma to construct pdf
    x = muhat-3*sigma:0.01:muhat+3*sigma;
    % plot PDF over histogram
    y = normpdf(x,muhat,sigma);
    
    new_y = (y ./ sum(y)) * 1000;   %%% why 1000?????????????/
    
    colors = rand(1, 3);
    plot(x,new_y,'Color', colors,'linewidth',2, 'DisplayName', cell2mat(allNames(k))); hold on;
    
    %% Try to plot the centers...
    %[counts, centers] = hist(plot_logged);
    %plot(centers, counts, 'o')
end
box off
xlabel('Log Length');
ylabel('Proportion of cellsNum Cells)');
legend('show');
%xlim([0 3]);

filename = strcat('Red Log Length per cell');
print(filename,'-dpng')
hold off;

%% Figure5:
figure;
ydata = [];
for i = 1:length(allTrialSheathsR)
    curData = allTrialSheathsR{i};
    if ~isempty(ydata)
        if isempty(curData)
            curData = zeros(1, length(ydata));   % creates array of zeros rather than be empty
        else
            [maxLength, ind]= max([length(curData), length(ydata)]);
            if ind == 2
                curData = padarray(curData, [0, length(ydata) - length(curData)], 'post');
            else
                ydata = padarray(ydata, [length(curData) - length(ydata), 0], 'post');
            end
        end
    end
    ydata = [ydata (curData)'];
end
ydata(ydata == 0) = nan;  % for plotting

cd(cur_dir);
UnivarScatter(ydata, 'Label', allNames, 'Whiskers', 'lines');
hold on;
title('Red Wrapping per cell');
ylabel('Num wrappings per cell');
ylim([0 25]);
cd(saveDir);

%Print figure
filename = strcat('Red Wrapping per cell');
print(filename,'-dpng')
hold off;


%% Figure 6:
% figure
% ydata = [];
% for i = 1:length(allTrialSheathsG)
%     curData = allTrialSheathsG{i};
%     if ~isempty(ydata)
%         if isempty(curData)
%             curData = zeros(1, length(ydata));   % creates array of zeros rather than be empty
%         else
%             [maxLength, ind]= max([length(curData), length(ydata)]);
%             if ind == 2
%                 curData = padarray(curData, [0, length(ydata) - length(curData)], 'post');
%             else
%                 ydata = padarray(ydata, [length(curData) - length(ydata), 0], 'post');
%             end
%         end
%     end
%     ydata = [ydata (curData)'];
% end
% ydata(ydata == 0) = nan;
% 
% cd('../');
% UnivarScatter(ydata, 'Label', allNames, 'Whiskers', 'lines');
% hold on;
% title('Green Wrapping per cell');
% ylabel('Num wrappings per cell');
% ylim([0 25]);
% 
% %Print figure
% cd(saveDir);
% filename = strcat('Green Wrapping per cell');
% print(filename,'-dpng')
% hold off;