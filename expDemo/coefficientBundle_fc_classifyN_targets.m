close all;
clear all;
clc;

fName = 'bundle_64_1024_1-2-3_1';
fNameModel = [fName '_model'];
load(fName);
load(fNameModel);

imgSet = unique(mod(data.input.dataClassificationGroup,100))';
sampleSigmaStep = [2];
sampleMultipliers = [0 4];
K = 5;

numTargets = length(imgSet) * length(sampleSigmaStep) * length(sampleMultipliers) * K * 2;

dataTargets.imgNumber = nan(numTargets, 1);
dataTargets.k = nan(numTargets, 1);
dataTargets.sample = nan(numTargets, 1);
dataTargets.step = nan(numTargets, 1);
dataTargets.expression = nan(numTargets, size(data.f.compomnentResponseStasitics,2));
dataTargets.expressionTarget = nan(numTargets, size(data.f.compomnentResponseStasitics,2));
dataTargets.img = nan(numTargets, 64, 64);
dataTargets.imgSpyr = nan(numTargets, length(data.f.coeffWeights));
dataTargets.phased = nan(numTargets, 1);

z = 0;
for i = 1:length(imgSet);
    %get set of images
    imgIndex = mod(data.input.dataClassificationGroup,100) == imgSet(i);
    imgTypeMask = data.input.dataClassificationGroup < 200;
    
    for k = 1:K
        for j = 1:length(sampleSigmaStep);
            %for this image and component, find nearest expression
            cMu = data.f.compomnentResponseStasitics(1,k);
            cSigma = data.f.compomnentResponseStasitics(2,k);
            cImg =  cMu + sampleSigmaStep(j)*cSigma;
                
            d = abs(squeeze(data.f.componentResponseMatrix(:, k)) - cImg);
            d = d .* imgIndex .* imgTypeMask;
            d(d == 0) = Inf;

            [~,di] = sort(d);

            imageIndex = di(1);
            
            for t = 1:length(sampleMultipliers);
                %construct component expression target
                expression = data.f.componentResponseMatrix(imageIndex,:);
                expressionTarget = expression;
                expressionTarget(k) = expressionTarget(k) * sampleMultipliers(t);
                
                %get expression of phase-scrambled image
                imageIndexPhased = imageIndex+size(data.input.dataImg,1);
                phasedExpression = data.f.componentResponseMatrix(imageIndexPhased,:);
                phasedExpressionTarget = phasedExpression;
                phasedExpressionTarget(k) = phasedExpressionTarget(k) * sampleMultipliers(t);
                
                %save image data
                z = z + 1;
                dataTargets.imgNumber(z) = imgSet(i);
                dataTargets.k(z) = k;
                dataTargets.sample(z) = sampleSigmaStep(j);
                dataTargets.step(z) = sampleMultipliers(t);
                dataTargets.expression(z,:) = data.f.componentResponseMatrix(imageIndex,:);
                dataTargets.expressionTarget(z,:) = expressionTarget;
                dataTargets.img(z,:,:) = single(data.input.dataImg(imageIndex,:,:));
                dataTargets.imgSpyr(z,:) = getSpyr(squeeze(dataTargets.img(z,:,:)));
                dataTargets.phased(z) = false;
                
                %save phase-scrambled data
                z = z + 1;
                dataTargets.imgNumber(z) = imgSet(i);
                dataTargets.k(z) = k;
                dataTargets.sample(z) = sampleSigmaStep(j);
                dataTargets.step(z) = sampleMultipliers(t);
                dataTargets.expression(z,:) = data.f.componentResponseMatrix(imageIndexPhased,:);
                dataTargets.expressionTarget(z,:) = phasedExpressionTarget;
                dataTargets.img(z,:,:) = single(data.input.dataScrambledImg(imageIndex,:,:));
                dataTargets.imgSpyr(z,:) = getSpyr(squeeze(dataTargets.img(z,:,:)));
                dataTargets.phased(z) = true;
                
                %save phase-scrambled data with unphased expression
                z = z + 1;
                dataTargets.imgNumber(z) = imgSet(i);
                dataTargets.k(z) = k;
                dataTargets.sample(z) = sampleSigmaStep(j);
                dataTargets.step(z) = sampleMultipliers(t);
                dataTargets.expression(z,:) = data.f.componentResponseMatrix(imageIndexPhased,:);
                dataTargets.expressionTarget(z,:) = expressionTarget;
                dataTargets.img(z,:,:) = single(data.input.dataScrambledImg(imageIndex,:,:));
                dataTargets.imgSpyr(z,:) = getSpyr(squeeze(dataTargets.img(z,:,:)));
                dataTargets.phased(z) = true;
                
            end
        end
    end
end

fNameTargets = [fName '_targets'];
save(fNameTargets, 'dataTargets');

%%
% copy to experiment directory
copyfile([fNameModel '.mat'], ['..\..\sPyrDerNet\data\' fNameModel '.mat']);
copyfile([fNameTargets '.mat'], ['..\..\sPyrDerNet\data\' fNameTargets '.mat']);


