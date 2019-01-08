%% Plot accuracy:

allTP = [];  allTN = []; allFP = []; allFN = []; allCells = [];

cd('./Accuracy_eval');
name = 'Ctr';
nameCat = strcat(name,'*.mat');
contestants=('QvP');
%Q=Qiao Lin, M=Matthew, P=Program
fnames = dir(nameCat);
numfids = length(fnames);

namecell=cell(length(fnames),1); %%must be out of function
for i=1:length(fnames) 
    namecell{i,1}=fnames(i).name ;
end
cd('../');
fnames=natsort(namecell);
cd('./Accuracy_eval');

numTruth_cells = 0;

for filenum = 1:3:numfids
    filename = fnames{filenum + 0};
    truth_DAPI = load(filename);
    truth_DAPI = truth_DAPI.ccount;
        
    numTruth_cells = numTruth_cells + length(truth_DAPI);  % ACTUAL NUM SHE COUNTED
   
    filename = fnames{filenum + 1};
    machine_DAPI = load(filename);
    machine_DAPI = machine_DAPI.wrappedDAPIR;
    
    filename = fnames{filenum + 2};
    objDAPI = load(filename);
    objDAPI = objDAPI.objDAPI;
    
    cd('../')
    [TP, TN, FP, FN] = accuracyEval(truth_DAPI, machine_DAPI, objDAPI);
    cd('./Accuracy_eval');
    
    allTP = [allTP TP];
    allTN = [allTN TN];
    allFP = [allFP FP];
    allFN = [allFN FN];
    allCells = [allCells length(objDAPI)];
end
cd('../');

total = (sum(allTP) + sum(allTN) + sum (allFP) + sum(allFN));
proportion_Truth = (sum(allTP) + sum(allFN))/ total

proportion_Truth_RAW = numTruth_cells/ total

proportion_Test = (sum(allTP) + sum(allFP))/ total
accuracy = (sum(allTP) + sum(allTN))./ total

% Also print out individual accuracy
accuracyIndividual = (allTP + allTN) ./ allCells;

clearvars -except allFN allFP allTN allTP nameCat accuracyIndividual accuracy name contestants total
x=strcat(name,'AccuracyData', contestants);
y=strrep(x,'*.mat','');

cd('./Accuracy_output_data');
save(x)
%Q=Qiao Lin, M=matthew, P=prgram


%Global Accuracy: 0.82308* 130 + 0.86879 * 282 + 0.82609 * 207 ==> 0.8449
