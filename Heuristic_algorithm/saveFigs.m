function [] = saveFigs(saveDirName, cur_dir, name, fileNum, objDAPI, wrappedDAPIR)

cd(saveDirName);

figure(67);
filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5), 'Ridges_PROCESS') ;
print(filename,'-dpng')
hold off;

figure(70);
filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5), 'unass_CORES') ;
print(filename,'-dpng')
hold off;

figure(71);
filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5), 'CORES') ;
print(filename,'-dpng')
hold off;

figure(24);
filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5), 'Ridges_Original') ;
print(filename,'-dpng'); hold off;

figure(102);
filename = strcat('Result', name, num2str((fileNum + 5 - 1)/5), 'cb_segment') ;
print(filename,'-dpng'); hold off;

cd(cur_dir); 

end