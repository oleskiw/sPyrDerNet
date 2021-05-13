close all;
clear all;
clc;

%this script will call the pytorch script to fit the droplet data
load expData

dirSelector = '_6o_1p6_dope';
inDir = ['~/dev/sPyrDerNet_data/input' dirSelector];
outDir = ['~/dev/sPyrDerNet_data/output' dirSelector];

FLAG_DO_INITIAL = false;
FLAG_DO_FINAL = true;

for i = 1:length(expData)
    fitInfoReg{i} = [0];
end
fitInfoReg{3} = [0.06 .08 .12];
fitInfoReg{5} = [0.12 .16 .20];
fitInfoReg{14} = [0.08 0.12 0.16];
fitInfoReg{15} = [0.02 0.03 0.04];
fitInfoReg{16} = [0.04 0.05 0.06];
fitInfoReg{17} = [0.08 0.12 0.16];
fitInfoReg{18} = [0.08 0.12 0.16];

fitInfoReg{19} = [0.12 0.14 0.16];
fitInfoReg{20} = [0.02 0.04 0.05];
fitInfoReg{21} = [0.04 0.06 0.08];
fitInfoReg{22} = [0.02 0.04 0.06];
fitInfoReg{23} = [0.06 0.08 0.12];

fitInfoReg{24} = [0.06 0.08 0.12];
fitInfoReg{25} = [0.06 0.08 0.12];
fitInfoReg{26} = [0.06 0.08 0.12];
fitInfoReg{27} = [0.06 0.08 0.12];
fitInfoReg{28} = [0.06 0.08 0.12];

%E = [3 5 14 15 16 17 18];
%E = [19 20 21 22 23];
E = [3 5 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28];
%E = 26;
%E = 1:size(expData,1);


%E1 = [17 18];
E1 = E;

%E2 = [3 5 14 15 16 17 18];
E2 = E;


%%
%for each cell, fit all lambda values with a few crossval partitions
if FLAG_DO_INITIAL
    for e = E1
        animalName = expData{e,2};
        unitName = expData{e,3};
        setName = expData{e,4};
        cellName = expData{e,6};

        %list desired lambdas for each cell
        cmdMain = 'source ~/py36/bin/activate; module load gcc python3/3.6.2 cuda/10.1.105_418.39; cd ~/dev/sPyrDerNet/; python main.py';
        %cmdOptions = sprintf('-smax 400 -smin 200 -l 0.01 0.02 0.04 0.06 0.08 0.12 0.16 -p 1 -f %s', e);
        cmdOptions = sprintf('-smax 1500 -smin 400 -l 0.01 0.02 0.04 0.06 0.08 0.12 0.16 0.20 -p 5 -f %s', [expData{e,6} '.mat']);
        tic;
        [status, cmdOut] = system([ cmdMain ' ' inDir ' ' outDir ' ' cmdOptions '; exit'], '-echo');
        t = toc;
        th = floor(t/(60^2));
        t = t - th*60^2;
        tm = floor(t/60);
        t = t-tm*60;
        ts = round(t);

        disp(['fit ' cellName ' (' cmdOptions sprintf(') in %02i:%02i:%02i', th, tm, ts)]);
    end
end


%%
%final pass
if FLAG_DO_FINAL
    for e = E2
        animalName = expData{e,2};
        unitName = expData{e,3};
        setName = expData{e,4};
        cellName = expData{e,6};
        
        lam = fitInfoReg{e};
        %list desired lambdas for each cell
        cmdMain = 'source ~/py36/bin/activate; module load gcc python3/3.6.2 cuda/10.1.105_418.39; cd ~/dev/sPyrDerNet/; python main.py';
        %cmdMain = 'cd "C:\Users\Oleskiw\Google Drive\wrx\Program\sPyrDerNet" & c:\python\Anaconda\envs\py36\python main.py ';
        cmdLambdas = ['-l' sprintf( ' %0.3f', fitInfoReg{e})];
        cmdOptions = sprintf('-smax 3000 -smin 1500 -p 0 %s -f %s', cmdLambdas, [expData{e,6} '.mat']);
        disp(cmdOptions);
        tic;
        [status, cmdOut] = system([ cmdMain ' ' inDir ' ' outDir '_final' ' ' cmdOptions '; exit'], '-echo');
        t = toc;
        th = floor(t/(60^2));
        t = t - th*60^2;
        tm = floor(t/60);
        t = t-tm*60;
        ts = round(t);

        disp(['fit ' cellName ' (' cmdOptions sprintf(') in %02i:%02i:%02i', th, tm, ts)]);
    end
end