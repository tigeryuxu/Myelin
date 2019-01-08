%% Load data
cd('./Accuracy_output_data');
nameCat = strcat('*.mat');      % CHANGE NAME
fnames = dir(nameCat);
numfids = length(fnames);

TP_A = [];
TN_A = [];
FP_A = [];
FN_A = [];

accu_A=[];
sens_A=[];
specif_A=[];
prec_A= [];
TPR_A = [];
FPR_A = [];
TNR_A = [];
FNR_A = [];
prop_A = [];

for filenum = 1:numfids
    filename = fnames(filenum).name;
    data = load(filename);
    
    TP = sum(data.allTP);
    TN = sum(data.allTN);
    FP = sum(data.allFP);
    FN = sum(data.allFN);
    
    TP_A = [TP_A; TP];
    TN_A = [TN_A; TN];
    FP_A = [FP_A; FP];
    FN_A = [FN_A; FN];
    
    
    total = TP + TN + FP + FN;
    accu =(TP + TN)/(total);
    sens =TP / (TP + FN);
    specif =TN / (TN + FP);
    prec = TP / (TP + FP);
    TPR = TP / (TP + FN);
    FPR = FP / (FP + TN);
    TNR = TN / (TN + FP);
    FNR = FN / (FN + TP);
    prop = (TP + FP) / (total)
    
    accu_A=[accu_A; accu];
    sens_A=[sens_A; sens];
    specif_A=[specif_A; specif];
    prec_A= [ prec_A; prec];
    TPR_A = [TPR_A; TPR];
    FPR_A = [FPR_A; FPR];
    TNR_A = [TNR_A; TNR];
    FNR_A = [FNR_A; FNR];
    
    prop_A = [prop_A; prop];
    
% Sensitivity = TP / TP + FN
% Specificity = TN / TN + FP
% Precision = TP / TP + FP
% True-Positive Rate = TP / TP + FN
% False-Positive Rate = FP / FP + TN
% True-Negative Rate = TN / TN + FP
% False-Negative Rate = FN / FN + TP
% 
% For good classifiers, TPR and TNR both should be nearer to 100%. Similar is the case with precision and accuracy parameters. On the contrary, FPR and FNR both should be as close to 0% as possible.
    
end
treatments= {'Control','Clem1','Clem2'};


%% Bar graph
data=[0.1650 prop_A(3) 0.1990 ; 0.4046 prop_A(1)  0.4194 ; 0.3404 prop_A(2) 0.3404];
figure;
bar(data);
set(gca,'xticklabel',treatments);
title('Proportion of Ensheathed Cells');
ylabel('Proportion(%)');
legend('Qiao Ling','Machine','Matthew');
ylim([0 1]);
 
filename = strcat('Graph compare counts');
print(filename,'-dpng')
hold off;


%% Make Table
treatRowNames = {'Clem1'; 'Clem2'; 'Control'};

%nameRows = {'Accuracy','Precision','Sensitity','Specificity','True-Positive','False-Positive','True-Negative','False-Negative'};
%T1 =table(prop_A, accu_A, sens_A, specif_A, prec_A, TPR_A, FPR_A, TNR_A, FNR_A,'RowNames', treatRowNames)
T1 =table(prop_A, accu_A, TPR_A, FPR_A, TNR_A, FNR_A,'RowNames', treatRowNames)


%bNames ={'TP','FP','TN','FN'};
T2 = table(TP_A, TN_A, FP_A, FN_A, 'Rownames', treatRowNames)



