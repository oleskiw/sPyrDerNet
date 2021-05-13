close all;
clear all;
clc;

addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\ExpoMatlab');
addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\ExpoMatlab\Helper functions');
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools');
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools\MEX')

FORCE_OPTIM_GEN = true;
VERBOSE = true;

dirSelector = '_multi_6o_3';

inputDir = ['C:\Users\Oleskiw\Google Drive\wrx\Program\sPyrDerNet_data\input' dirSelector ];
dataDir = 'C:\Users\Oleskiw\Google Drive\wrx\Program\V2DropletExperiment\Data\';
parsedDir = [dataDir 'parsed' dirSelector '\'];
figDir = [parsedDir 'figs\' ];

load expData;

presentationDuration = 0.200;
crossValFold = 3;
crossValHighRep = true;

%E = [20];
%E = [18 20 21 22];
%E = [3 5 14 15 16 17 18 19 20 21 22];
%E = 1:length(expData);

E = [991];

MULTI{991}.animalName = 'm685';
MULTI{991}.unitName = 'p1#6';
MULTI{991}.setName = 'R_ExpA_sf1000_d20_n256';

mkdir(inputDir)
for e = E
    PARSE_MULTI = false;
    if isfield(MULTI{e}, 'animalName')
        PARSE_MULTI = true;
        animalName = MULTI{e}.animalName;
        unitName = MULTI{e}.unitName;
        setName = MULTI{e}.setName;
    else
        animalName = expData{e,2};
        unitName = expData{e,3};
        setName = expData{e,4};
    end

    filename = [animalName unitName setName];
    load([parsedDir filename]);
   
    %determine channels worth processing
    disp('Determining viable contacts...');
    contactModulated = (([rec.info.responseMetric.stimAveragePeak] - [rec.info.responseMetric.baseline])./[rec.info.responseMetric.baseline]) >= 1.5;
    contactDriven = ([rec.info.responseMetric.stimAveragePeak]) > 1.5;
    contactPreferential = ([rec.info.responseMetric.stimPrefPeak] - [rec.info.responseMetric.baseline]) > 8;
    rec.info.contactProcessed = (contactModulated & contactDriven & contactPreferential);
    
    rec.info.contactProcessed = rec.info.contactProcessed(rec.info.contactProcessed < 200);
    %% compute STAs
    disp('Computing STAs...');

    pyr = [rec.info.Pyr.scales rec.info.Pyr.orientations rec.info.Pyr.resolution];
    pind = rec.info.Pyr.pind;

    rec.info.StaFit.outputPolynomialOrder = 2;
    rec.info.StaFit.crossvalPartitions = 10;

    %afferant transfer function
    PFN = rec.info.StaFit.outputPolynomialOrder;
    CROSSVAL_PARTITIONS = rec.info.StaFit.crossvalPartitions;

    %compute STAs
    staImg = nan(length(rec.info.contact(rec.info.contactProcessed)), length(rec.info.stimImgList(1,:)));
    staImgPower = nan(length(rec.info.contact(rec.info.contactProcessed)), length(rec.info.stimImgList(1,:)));
    staSpyr = nan(length(rec.info.contact(rec.info.contactProcessed)), length(rec.info.stimSpyrList(1,:)));
    staSpyrPower = nan(length(rec.info.contact(rec.info.contactProcessed)), length(rec.info.stimSpyrList(1,:)));

    
    
    %generate STAs to all data for visualization
    responseVector = rec.info.responseList(rec.info.contactProcessed,:);
    for z = 1:2
        switch z
            case 1
                D = rec.info.stimImgList;
            case 2
                D = rec.info.stimSpyrList;
        end

        %compute stas for every contact
        stimSta = responseVector * D;
        stimSta = stimSta ./ repmat(sum((responseVector),2), [1,size(D,2)]);
        stimAll = sum(D,1);
        stimAll = stimAll / size(responseVector,2);
        stimSta = (stimSta ./ repmat(stimAll, [size(stimSta,1),1])) - 1;
        stimSta = stimSta .* repmat(abs(stimAll), [size(stimSta,1),1]);
        stimSta(isnan(stimSta)) = 0;

        stimStaPower = responseVector * abs(D);
        stimStaPower = stimStaPower ./ repmat(sum((responseVector), 2), [1,size(D,2)]);
        stimAllPower = sum(abs(D),1);
        stimAllPower = stimAllPower / size(responseVector,2);
        stimStaPower = (stimStaPower ./ repmat(stimAllPower,[size(stimSta,1),1])) - 1;
        stimStaPower = stimStaPower .* repmat(abs(stimAllPower), [size(stimSta,1),1]);
        stimStaPower(isnan(stimStaPower)) = 0;

        switch z
            case 1
                staImg(:) = stimSta;
                staImgPower(:) = stimStaPower;
            case 2
                staSpyr(:) = stimSta;
                staSpyrPower(:) = stimStaPower;
        end
    end

    rec.info.staImg = staImg;
    rec.info.staImgPower = staImgPower;
    rec.info.staSpyr = staSpyr;
    rec.info.staSpyrPower = staSpyrPower;
    
    %for every contact to be processed
    contactProcessed = rec.info.contact(rec.info.contactProcessed);
    for c = 1:length(contactProcessed);
        ci = find(rec.info.contact == contactProcessed(c));
        
        responseVector = rec.info.responseList(ci,:);
        %perform epic crossval
        %create parititon vector
        cvp = CROSSVAL_PARTITIONS;
        cvParts = repmat((1:cvp), [1 ceil(length(responseVector)/cvp)]);
        cvParts = cvParts(1:length(responseVector));
        cvParts = cvParts(randperm(length(cvParts)));

        %for every partition=
        staFitPredictions = nan(size(responseVector));

        for i = 1:cvp
            train = cvParts ~= i;

            %build sta
            stimSta = responseVector(train) * rec.info.stimSpyrList(train,:);
            stimAll = sum(rec.info.stimSpyrList(train,:), 1);
            stimSta = stimSta / sum(responseVector(train));
            stimAll = stimAll / length(responseVector(train));

            stimSta = (stimSta ./ stimAllPower) - 1;
            stimSta = stimSta .* abs(stimAllPower);
            stimSta(isnan(stimSta)) = 0;

            %fit output nonlin
            staPred = stimSta * rec.info.stimSpyrList(train,:)';
            pfitMu = mean(staPred);
            pfitSigma = std(staPred);
            pfitPoly = polyfit((staPred - pfitMu)/pfitSigma, responseVector(train), PFN);
            star = polyval(pfitPoly, (staPred - pfitMu)/pfitSigma);

            %predict partition
            testProj = stimSta * rec.info.stimSpyrList(~train,:)';
            testPred =  polyval(pfitPoly, (testProj-pfitMu)/pfitSigma);
            staFitPredictions(~train) = testPred;

        end

        clf;
        plot(staFitPredictions, responseVector, '.');
        ve = var(staFitPredictions - responseVector) / var(responseVector);
        fprintf('-sPyr power sta explained variance: %0.3f\n', (1-ve));

        if VERBOSE
            close all;
            h = figure('position', [10         662        1938         583]);
            colormap(gray(256));

            imgSta = reshape(rec.info.staImg(c,:) * rec.info.Ds.MdsI', rec.info.Pyr.resolution*[1 1]);
            imgSta(isnan(imgSta)) = 0;
            imgStaPower = reshape(rec.info.staImgPower(c,:) * rec.info.Ds.MdsI', rec.info.Pyr.resolution*[1 1]);
            imgStaPower(isnan(imgStaPower)) = 0;

            nFactorImg = max(abs([imgSta(:); imgStaPower(:)]));

            staSpyrFull = rec.info.staSpyr(c,:) * rec.info.Ds.Mds';
            staSpyrFull = staSpyrFull / max(abs(staSpyrFull(:)));

            staSpyrOn = staSpyrFull;
            staSpyrOn(sum(prod(pind,2))+1:end) = 0;
            staSpyrOff = staSpyrFull;
            staSpyrOff(1:sum(prod(pind,2))) = 0;

            subplot(2,3,1);
            image(imgSta*(128/nFactorImg)+128);
            axis equal tight;
            set(gca, 'xTick', [], 'yTick', []);
            xlabel('Image STA');

            subplot(2,3,2);
            image(imgStaPower*(128/nFactorImg)+128);
            set(gca, 'xTick', [], 'yTick', []);
            axis equal tight;
            xlabel('Image Power STA');

            subplot(2,3,4);
            spyrDisp3(staSpyrOn, pind, pyr, true, 0, rec.expo.stimDiameter(ci));
            title('Spyr On STA');

            subplot(2,3,5);
            spyrDisp3(staSpyrOff, pind, pyr, true, 0, rec.expo.stimDiameter(ci));
            title('Spyr Off STA');

            subplot(2,3,6);
            spyrDisp3(staSpyrFull, pind, pyr, true, 0, rec.expo.stimDiameter(ci));
            title('Spyr STA');

            if PARSE_MULTI 
                pdfname = [animalName unitName '_' sprintf('%03i', rec.info.contact(ci))];
            else
                pdfname = expData{e,6};
            end
            title(['Cell: ' pdfname], 'interpreter', 'none');
            set(h,'Units','Inches');
            pos = get(h,'Position');
            set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            saveas(h, [figDir 'sta_' pdfname '.pdf'], 'pdf');
        end

        h = figure('position', [50 50 1200 600]);
        %project STA onto stimuli
        resp = rec.info.responseList(ci,:);

        staSpyrFull = rec.info.staSpyr(c,:);
        staSpyrOn = staSpyrFull;
        staSpyrOn((length(staSpyrOn)/2)+1:end) = 0; 
        staSpyrOff = staSpyrFull;
        staSpyrOff(1:(length(staSpyrOff)/2)) = 0; 

        sli = floor(length(staSpyrOn)/2);
        staLinear = rec.info.staSpyr(c,1:sli) - rec.info.staSpyr(c,sli+1:end);

        subplot(2,4,1);
        linearData = (rec.info.stimSpyrList(:,1:sli)-rec.info.stimSpyrList(:,sli+1:end));
        staSpyrScaledLinear = staLinear * (resp * (staLinear * linearData')');
        staPred = staSpyrScaledLinear * linearData';
        [p0fit, S0, mu0] = polyfit(staPred, resp, PFN);
        scatter(staPred, resp, 'k'); hold on;
        plot(sort(staPred), polyval(p0fit, sort(staPred), S0, mu0), 'r-');
        tmp = corrcoef(resp,staPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Linear projection'); ylabel('Response (spk/sec)');

        subplot(2,4,2);
        staSpyrScaledOn = staSpyrOn * (resp * (staSpyrOn * rec.info.stimSpyrList')');
        staPred = staSpyrScaledOn * rec.info.stimSpyrList';
        [p1fit, S1, mu1] = polyfit(staPred, resp, PFN);
        scatter(staPred, resp, 'k'); hold on;
        plot(sort(staPred), polyval(p1fit, sort(staPred), S1, mu1), 'r-');
        tmp = corrcoef(resp,staPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr On projection'); ylabel('Response (spk/sec)');

        subplot(2,4,3);
        staSpyrScaledOff = staSpyrOff * (resp * (staSpyrOff * rec.info.stimSpyrList')');
        staPred = staSpyrScaledOff * rec.info.stimSpyrList';
        [p2fit, S2, mu2] = polyfit(staPred, resp, PFN);
        scatter(staPred, resp, 'k'); hold on;
        plot(sort(staPred), polyval(p2fit, sort(staPred), S2, mu2), 'r-');
        tmp = corrcoef(resp,staPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Off projection'); ylabel('Response (spk/sec)');

        subplot(2,4,4);
        staSpyrScaledFull = staSpyrFull * (resp * (staSpyrFull * rec.info.stimSpyrList')');
        staPred = staSpyrScaledFull * rec.info.stimSpyrList';
        [p3fit, S3, mu3] = polyfit(staPred, resp, PFN);
        scatter(staPred, resp, 'k'); hold on;
        plot(sort(staPred), polyval(p3fit, sort(staPred), S3, mu3), 'r-');
        tmp = corrcoef(resp,staPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Full projection'); ylabel('Response (spk/sec)');

        subplot(2,4,5);
        staPred =  staSpyrScaledLinear * linearData';
        staFitPred = polyval(p0fit,staPred, S0, mu0);
        plot([min(resp) max(resp)], [min(resp) max(resp)], 'k:'); hold on;
        scatter(staFitPred, resp, 'k', 'MarkerEdgeColor', [.3 .3 .3]);
        tmp = corrcoef(resp,staFitPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Linear linear fit'); ylabel('Response (spk/sec)');

        subplot(2,4,6);
        staPred = staSpyrScaledOn * rec.info.stimSpyrList';
        staFitPred = polyval(p1fit,staPred, S1, mu1);
        plot([min(resp) max(resp)], [min(resp) max(resp)], 'k:'); hold on;
        scatter(staFitPred, resp, 'k', 'MarkerEdgeColor', [.3 .9 .3]);
        tmp = corrcoef(resp,staFitPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr On linear fit'); ylabel('Response (spk/sec)');

        subplot(2,4,7);
        staPred = staSpyrScaledOff * rec.info.stimSpyrList';
        staFitPred = polyval(p2fit,staPred, S2, mu2);
        plot([min(resp) max(resp)], [min(resp) max(resp)], 'k:'); hold on;
        scatter(staFitPred, resp, 'k', 'MarkerEdgeColor', [.9 .3 .3]);
        tmp = corrcoef(resp,staFitPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Off linear fit'); ylabel('Response (spk/sec)');

        subplot(2,4,8);
        staPred = staSpyrScaledFull * rec.info.stimSpyrList';
        staFitPred = polyval(p3fit,staPred, S3, mu3);
        plot([min(resp) max(resp)], [min(resp) max(resp)], 'k:'); hold on;
        scatter(staFitPred, resp, 'k', 'MarkerEdgeColor', [.9 .9 .3]);
        tmp = corrcoef(resp,staFitPred);
        title(['r = ' num2str(tmp(1,2))])
        xlabel('sPyr Full linear fit'); ylabel('Response (spk/sec)');

        title(['r = ' num2str(tmp(1,2)) ', Cell: ' pdfname], 'interpreter', 'none');
        set(h,'Units','Inches');
        pos = get(h,'Position');
        set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        saveas(h, [figDir 'fit_' pdfname '.pdf'], 'pdf');

        %store sta fit info
        staRf(c).rfEstimate = staSpyrScaledFull;
        staRf(c).fit.p = p3fit;
        staRf(c).fit.s = S3;
        staRf(c).fit.mu = mu3;
        

        %% estimate noise ceiling
        r = randi(40,[3 1000]);
        r = mean(r,1);
        robs = normrnd(r, sqrt(r));
        robs = max(robs,0);

        %check for repeats
        stimWithRepeats = rec.info.responseRepList(ci,:) > 3;
        rates = rec.info.responseList(ci,stimWithRepeats);
        ratesStd = rec.info.responseStdList(ci,stimWithRepeats);
        repFraction = (sum(stimWithRepeats) / length(stimWithRepeats));

        %calculate CV
        responseAverageCV = mean(ratesStd/rates);

        if ~isfield(rec.info, 'dataOriSf')
            estCV = responseAverageCV;
        else
            if mean(rec.info.responseRepList(ci,:)) > mean(rec.info.dataOriSf.rateReps(:))
                estCV = responseAverageCV;
            else
                oriSfCV = rec.info.dataOriSf.rateVar.^(1/2) ./ rec.info.dataOriSf.rateMean;
                oriSfAverageCV = mean(oriSfCV(rec.info.dataOriSf.rateReps>=3 & rec.info.dataOriSf.rateMean ~= 0));
                estCV = oriSfAverageCV;
            end
        end

        %noise ceiling bootstrap
        reps = 10*CROSSVAL_PARTITIONS;

        noiseEV(c).est = nan(1, reps);
        for i = 1:length(noiseEV(c).est)
            simRates = 1:length(rec.info.responseList(ci,:));
            for j = 1:length(rec.info.responseList(ci,:))
                simRates(j) = mean( normrnd(rec.info.responseList(ci,j) + rec.info.baselineRate(ci), abs((rec.info.responseList(ci,j) + rec.info.baselineRate(ci))) * estCV, [1 rec.info.responseRepList(ci,j)]) );
            end
            simRates = simRates - rec.info.baselineRate(ci);

            noiseEV(c).est(i) = 1-var( (rec.info.responseList(ci,:) - simRates) ) / var(simRates);
        end
        noiseEV(c).stat = [mean(noiseEV(c).est), std(noiseEV(c).est)];
        fprintf('-noise ceiling estimate: %0.3f +- %.2f \n', mean(noiseEV(c).est), std(noiseEV(c).est));
    end
    
    %% aggregate plots
    disp('Making sta PDF...');
    staFilename = 'archive_sta.pdf';
    cmdToRun = ['"C:\Program Files (x86)\PDFtk Server\bin\pdftk.exe" sta*.pdf output ' staFilename];
    system(['cd "' figDir '"' '& ' cmdToRun]);
    disp(['Wrote to ' staFilename]);    
    
    disp('Making processed psth PDF...');
    fitFilename = 'archive_processed_psth.pdf';
    ppsth = [];
    for c = 1:length(contactProcessed);
        ci = find(rec.info.contact == contactProcessed(c));
        
        ppsth = [ppsth 'psth_' animalName unitName '_' sprintf('%03i', rec.info.contact(ci)) '.pdf '];
    end
    
    cmdToRun = ['"C:\Program Files (x86)\PDFtk Server\bin\pdftk.exe" ' ppsth ' output ' fitFilename];
    system(['cd "' figDir '"' '& ' cmdToRun]);
    disp(['Wrote to ' fitFilename]);


    %% generate regularization scaling vector
    regWeightUniform = ones(1,sum(prod(pind,2)));
    regWeight = nan(1,sum(prod(pind,2)));

    idx = 1;
    topScale = pind(1,1);
    for p = 1:size(pind,1);        
        w = ones(1,prod(pind(p,:)));
        regWeight(idx:(idx+length(w)-1)) = w / length(w);
        idx = idx + length(w);
    end
    
    regWeight = regWeight / sum(regWeight) * sum(regWeightUniform);
    regWeight = [1*regWeight 1*regWeight];
    rw = regWeight * rec.info.Ds.Mds * rec.info.Ds.Mds';
    
    %[m, mi] = max(rec.info.Ds.Mds,[],1);
    %regWeightDs = regWeight(mi);
    %rw = regWeightDs * rec.info.Ds.Mds';
    
    regularizationWeights = regWeight * rec.info.Ds.Mds;
    
%     %display for diagnostics
%     spyrDisp3(rw/max(rw(:)), pind, pyr, false);
    
    %% collect data for fitting
    if ~exist([inputDir '\' filename '.mat'], 'file') || FORCE_OPTIM_GEN
        fprintf('Generating optimization data...\n');

        %store responses and stimuli
        responseWeight = @(x,a) 1+ (x  / max(x) * (a-1));
        responseWeightPoisson = @(x,a) 1./(abs(x).^(0.5)+a);
        rec.n.response = rec.info.responseList(rec.info.contactProcessed,:);
        rec.n.responseWeights = responseWeightPoisson(rec.info.responseList(rec.info.contactProcessed,:)' + repmat(rec.info.baselineRate(rec.info.contact(rec.info.contactProcessed)), [size(rec.info.responseList,2) 1]), 1);
        rec.n.stimSpyr = rec.info.stimSpyrList;
        
        cvf = crossValFold;
        crossValPart = repmat((1:cvf), [1 ceil(size(rec.n.response,2)/cvf)]);
        rng(e);
        if crossValHighRep
            partIds = rec.info.responseRepList(ci,:) > median(rec.info.responseRepList(ci,:));
            crossValPart(~partIds) = max(crossValPart)+1;
            sec = crossValPart(partIds);
            crossValPart(partIds) = sec(randperm(length(sec)));
        else
            %validation cases, assign each stim/resp to a cvf partition number
            crossValPart = crossValPart(1:length(rec.n.response));
            crossValPart = crossValPart(randperm(length(crossValPart)));
        end
        rec.n.crossValPart = crossValPart;

        %store sta rf
        rec.n.staRf = staRf;

        %downsample matrices
        rec.n.Mds.Mds = (rec.info.Ds.Mds);
        rec.n.Mds.MdsI = (rec.info.Ds.MdsI);
        %rec.n.Mx = full(rec.info.Ds.Mds' * rec.info.Ds.Ax * rec.info.Ds.Mds);
        %rec.n.My = full(rec.info.Ds.Mds' * rec.info.Ds.Ay * rec.info.Ds.Mds);

        %set initialization conditions
        rec.n.regWeight = repmat(regularizationWeights, [size(rec.n.response,1) 1]);
        rec.n.initRf = repmat(((rand(1, size(rec.n.stimSpyr,2)) - 1/2) * .01), [size(rec.n.response,1) 1]);
        rec.n.initIn = repmat(.0001, [size(rec.n.response,1) 1]);
        rec.n.initOut = repmat([0 1 -1], [size(rec.n.response,1) 1]);

        % rec.n.initRf = rec.n.rfEstimate;
        % rec.n.initIn = epsilon;
        % rec.n.initOut = fliplr(p2fit);

        rec.n.noiseEV = noiseEV;
        rec.n.Pyr = rec.info.Pyr;
        if isfield(rec.info, 'dataOriSf')
            rec.n.dataOriSf = rec.info.dataOriSf;
        end
        if isfield(rec.info, 'dataTexture')
            rec.n.dataTexture = rec.info.dataTexture;
        end
        rec.n.stimDiameter = rec.expo.stimDiameter;
        rec.n.rfDiameter = rec.expo.rfDiameter;

        if PARSE_MULTI 
            outname = [animalName unitName '_' sprintf('C%i', length(rec.info.contact(rec.info.contactProcessed)))];
        else
            outname = expData{e,6};
        end
        
        %save neuron data
        clear neuronData
        neuronData = rec.n;
        save([inputDir '\' outname '.mat'], 'neuronData', '-v7.3');
    end

end




