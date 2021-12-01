addpath('./lib/');
addpath('./datasets/');

clear;
DataName={'brain'; 'breast3'; 'jaffe'; 'lung'; 'mnist'; 'nci'; 'ORL'; 'Palm'};
for i=1:size(DataName,1)
     dataset = DataName{i};
     f = runNAUFS(dataset);
end


