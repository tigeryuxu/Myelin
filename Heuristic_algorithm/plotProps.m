function [normProp] = plotProps(allNames, allWrapped, allSumO4, allTotalCells, name, control_idx, saveDirName)

    cd(saveDirName);
    c = categorical(allNames);
    allProp = allWrapped./allTotalCells;
    allO4_Prop = allWrapped./allSumO4;
    
    %%plot proportion
    figure;
    subplot(2,3,1)
    
    normProp = allO4_Prop ./ allO4_Prop(control_idx);
    bar(c, normProp);
    ylim([0 5]);
    title('Norm prop W_O4 to C');
    hold on;
    
    % Plot proportion of O4 that is wrapped
    subplot(2,3,2)
    bar(c, allO4_Prop);
     str = strcat('Prop ', 'wrapped/', name);
    title(str);
    ylim([0 1]);
    hold on;
    
    %%plot proportion
    subplot(2,3,3)
    bar(c, allProp);
    ylim([0 1]);
    title('Prop wrapped/total');
    hold on;
    
    % Plot number that are O4
    subplot(2,3,4)
    bar(c, allSumO4./allTotalCells);
    str = strcat('Prop ', name, '/total');
    title(str);
    ylim([0 1]);
    hold on;
    
    %%plot number
    subplot(2,3,5)
    bar(c, allWrapped);
    str = strcat('Total cells wrap', name);
    title(str);
    ylim([0 500]);
    hold on;
    
    %plot total counted
    subplot(2,3,6)
    bar(c, allTotalCells);
    ylim([0 1500]);
    title('Total cells counted');
    hold on;
    
    %print figure
    filename = strcat('Graph stats ', name);
    print(filename,'-dpng')
    hold off;
    

end