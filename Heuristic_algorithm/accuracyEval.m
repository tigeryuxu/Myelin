function [truePos, trueNeg, falsePos, falseNeg] = accuracyEval(truth_DAPI, machine_DAPI, allDAPI)

truePos = 0; trueNeg = 0;
falsePos = 0; falseNeg = 0;

tmp = zeros(1460, 1936);
tmpTruth = zeros(1460, 1936);
tmpM = zeros(1460, 1936);

for k = 1:length(allDAPI)
    curDAPI = allDAPI{k};
    
    tmp(curDAPI) = 1;
    %figure(1); imshow(tmp);
    
    truth = 0;
    for i = 1:length(truth_DAPI)
        if ~isempty(truth_DAPI{i})
            curTruth = sub2ind([1460 1936], floor(truth_DAPI{i}(2)), floor(truth_DAPI{i}(1)));
            same = ismember(curDAPI, curTruth);  % find the identical value
            
            tmpTruth(curTruth) = 1;
            %figure(2); imshow(tmpTruth);
            
            if ~isempty(find(same, 1))
                truth = 1;
                break;
            end
        end
    end
    
    machine = 0;
    for j = 1:length(machine_DAPI)
        if ~isempty(machine_DAPI{j})
            curMachine = sub2ind([1460 1936], floor(machine_DAPI{j}(2)), floor(machine_DAPI{j}(1)));
            
            %curMachine = machine_DAPI{j};
            same = ismember(curDAPI, curMachine);  % find the identical value
           
            tmpM(curMachine) = 1;
            
            %figure(3); imshow(tmpM);
            %same = ismember(machine_DAPI, curMachine);  % find the identical value
            if ~isempty(find(same, 1))
                machine = 1;
                break;
            end
        end
    end
    
    % TP:
    if truth && machine
        truePos = truePos + 1;
        % FP
    elseif ~truth && machine
        falsePos = falsePos + 1;
        % FN
    elseif truth && ~machine
        falseNeg = falseNeg + 1;
        % TN
    elseif ~truth && ~machine
        trueNeg = trueNeg + 1;
    end
end
end