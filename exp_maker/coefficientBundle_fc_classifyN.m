%general coefficient bundler, trains a set of linear components over
%arbitrary nonlinear transformation of steerable pyramid coeffieints
%computed from image patches and phase-scrambled counterparts
close all;
clear all;
clc;

addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools');
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools\MEX')
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\FastICA_25');
addpath('C:\Users\Oleskiw\Documents\GitHub\textureSynth');
addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\Toolbox');

RESIZE_FACTOR = .4;     %image rescaling factor
N = 64;                 %patch size

M = 2^10;               %total number of patches to consider
K2 = 2^8;               %random patches for distribution plots
FSET = [1 2 3];             %set of images to process

I = 2^5;                %coefficients used for classification
DZ = 7;                 %recon component size
DR = 2;                 %recon component count

PSK = 2^4;              %bootstrap count for projection statistics

%prepare set of coefficient transfer functionselse
%coeffTransferSet = {@(x)x, @(x)x.^2};
%coeffTransferSet = {@(x)x, @(x)abs(x)};
%coeffTransferSet = {@(x)abs(x), @(x)x.^2};
%coeffTransferSet = {@(x)exp(x), @(x)exp(-x)};
%coeffTransferSet = {@(x)abs(x)};
%coeffTransferSet = {@(x)x};
coeffTransferSet = {@(x)(x.^2 + .001).^(1/2)};


%turn on to enable texture statistic analysis and fitting
FLAG_DO_TEXTURESTATS  = false ;

J = length(coeffTransferSet);

%prepare set of files
F = length(FSET);
imgFilenameN = {'images\plant01.jpg',...
                'images\bark01.jpg',...
                'images\grass01.jpg',...
                'images\bark02.jpg',...
                'images\earth01.jpg',...
                'images\stone01.jpg',...
                'images\grass03.jpg',...
                'images\earth02.jpg',...
                'images\grass02.jpg',...
                'images\metal01.jpg',...
                'images\skin01.jpg',...
                'images\stone02.jpg'};
imgFilenameN = imgFilenameN(FSET);
            
SIGMAS_INNER = 3*[0 2 4];
SIGMAS_OUTER = 4*[0 2 4];

CLASSIFICATION_VALIDATION_BOOTSTRAP = 64;
CLASSIFICATION_VALIDATION_SIZE = 0.05;

K = floor(M/F);         %patches per image

%%
if ~exist('DMcoeff', 'var')
    %construct pyramid bundle weight index
    coeffWeights = [];
    s = (N)/2^3;
    for si = 1:3;
        if si == 1
            j = 1;
        elseif si == 2;
            j = 4;
        elseif si == 3;
            j = 5;
        end

        for ji = 1:j
            [cmc, cmr] = meshgrid(1:s*2, 1:s*2);
            %cm = exp(-((cmc-(s+1))/(SIGMAS(si))).^2) .* exp(-((cmr-(s+1))/(SIGMAS(si))).^2);
            r = sqrt((cmc-(s)).^2 + (cmr-(s)).^2);
            cm = zeros(size(r));
            cm(r<SIGMAS_INNER(si)) = 1 ;
            if SIGMAS_INNER(si) > 0
                cm(r>=SIGMAS_INNER(si)& r <= SIGMAS_OUTER(si)) = (-r(r>=SIGMAS_INNER(si)& r <= SIGMAS_OUTER(si)) + SIGMAS_OUTER(si))/(SIGMAS_OUTER(si)-SIGMAS_INNER(si));
                cm(r>SIGMAS_OUTER(si)) = 0;
            end

            %special case for residuals, turn off or on by cm*0
            if ji == 5
                coeffWeights = [cm(:)*0; coeffWeights];
            else
                coeffWeights = [cm(:); coeffWeights];
            end
        end

        s = s * 2;
    end

    %cut off weights below threshold
    coeffWeights(coeffWeights < .01) = 0;
    coeffIndex = coeffWeights > 0;
    paramCount = sum(coeffIndex);

    %data matrices
    dataImg = nan(K*F,N,N);
    dataScrambledImg = nan(K*F,N,N);
    dataMatrix = nan(K*F,J*paramCount);
    dataScrambledMatrix = nan(K*F,J*paramCount);
    dataClassificationGroup = nan(K*F,1);
    dataScrambledClassificationGroup = nan(K*F,1);
    dataTexture(K*F) = textureAnalysis(zeros(64,64), 4, 4, 9);
    dataScrambledTexture = textureAnalysis(zeros(64,64), 4, 4, 9);
    dataTextureStat = nan(K*F,10);
    dataScrambledTextureStat = nan(K*F,10);
    dataTextureStatVec = nan(K*F,755);
    dataScrambledTextureStatVec  = nan(K*F,755);

    %for every image, process
    kSave = 4;
    imgSample = zeros(F,2,256,256);
    imgSampleSmall = zeros(kSave,F,2,N,N);
    imgSampleWeighted = zeros(kSave,F,2,N,N);
    
    fprintf('Processing Images:');
    for f = 1:F
        fprintf('\n %s ', imgFilenameN{f});
        imgFilename = imgFilenameN{f};

        %load and desaturate image
        im = imread(imgFilename);
        im = hsv2rgb(rgb2hsv(im).*repmat(reshape([1,0,1],[1 1 3]), [size(im,1) size(im,2), 1]));
        im = squeeze(im(:,:,1));

        %rescale image
        im = imresize(im, RESIZE_FACTOR);

        %make phase-scrambled version;
        fim = fft2(im);
        fimPower = abs(fim);
        fimRandPhase = rand(size(fim))*2*pi;
        imRandPhase = abs(ifft2(fimPower.*exp(1i.*fimRandPhase)));

        for k = 1:K
            if mod(log2(k),1) == 0
                fprintf('.');
            end
            %isolate a patch
            rr = randi(size(im,1)-2*N);
            rc = randi(size(im,2)-2*N);
            imPatch = im(rr:rr+N-1,rc:rc+N-1,1);
            rr = randi(size(imRandPhase,1)-2*N);
            rc = randi(size(imRandPhase,2)-2*N);
            imScrambledPatch = imRandPhase(rr:rr+N-1,rc:rc+N-1,1);

            %remove mean and normalize variance
            imPatch = imPatch - mean(imPatch(:));
            imPatch = imPatch/sqrt(var(imPatch(:)));
            imScrambledPatch = imScrambledPatch - mean(imScrambledPatch(:));
            imScrambledPatch = imScrambledPatch/sqrt(var(imScrambledPatch(:)));

            %build pyramid
            filts = 'sp3Filters';
            edges = 'reflect1';
            [imSpyr, imPind] = getSpyr(imPatch);
            [imScrambledSpyr, ~] = getSpyr(imScrambledPatch);

            zeroImSpyr = zeros(length(imSpyr),1);

            %save image patch samples; 
            if k <= kSave;
                imgSampleSmall(k,f,1,:,:) = imPatch .* cm;
                imgSampleSmall(k,f,2,:,:) = imScrambledPatch .* cm;

                c = zeros(length(coeffIndex),1);
                c(coeffIndex) = imSpyr(coeffIndex).*coeffWeights(coeffIndex);
                imWeighted = putSpyr(c, imPind);
                imgSampleWeighted(k,f,1,:,:) = imWeighted;

                c = zeros(length(coeffIndex),1);
                c(coeffIndex) = imScrambledSpyr(coeffIndex).*coeffWeights(coeffIndex);
                imScrambledWeighted = putSpyr(c, imPind);
                imgSampleWeighted(k,f,2,:,:) = imScrambledWeighted;
            end
            
            %store patch statistics
            DT = textureAnalysis(imWeighted, 4, 4, 9);
            dataTexture(((f-1)*K)+k) = DT;
            DST = textureAnalysis(imScrambledWeighted, 4, 4, 9);
            dataScrambledTexture(((f-1)*K)+k) = DST;
            
            dataTextureStat(((f-1)*K)+k,1) = norm(DT.pixelStats);
            dataTextureStat(((f-1)*K)+k,2) = norm(DT.pixelLPStats);
            dataTextureStat(((f-1)*K)+k,3) = norm(DT.autoCorrReal(:));
            dataTextureStat(((f-1)*K)+k,4) = norm(DT.autoCorrMag(:));
            dataTextureStat(((f-1)*K)+k,5) = norm(DT.magMeans);
            dataTextureStat(((f-1)*K)+k,6) = norm(DT.cousinMagCorr(:));
            dataTextureStat(((f-1)*K)+k,7) = norm(DT.parentMagCorr(:));
            dataTextureStat(((f-1)*K)+k,8) = norm(DT.cousinRealCorr(:));
            dataTextureStat(((f-1)*K)+k,9) = norm(DT.parentRealCorr(:));
            dataTextureStat(((f-1)*K)+k,10) = norm(DT.varianceHPR);
            
            dataScrambledTextureStat(((f-1)*K)+k,1) = norm(DST.pixelStats);
            dataScrambledTextureStat(((f-1)*K)+k,2) = norm(DST.pixelLPStats);
            dataScrambledTextureStat(((f-1)*K)+k,3) = norm(DST.autoCorrReal(:));
            dataScrambledTextureStat(((f-1)*K)+k,4) = norm(DST.autoCorrMag(:));
            dataScrambledTextureStat(((f-1)*K)+k,5) = norm(DST.magMeans);
            dataScrambledTextureStat(((f-1)*K)+k,6) = norm(DST.cousinMagCorr(:));
            dataScrambledTextureStat(((f-1)*K)+k,7) = norm(DST.parentMagCorr(:));
            dataScrambledTextureStat(((f-1)*K)+k,8) = norm(DST.cousinRealCorr(:));
            dataScrambledTextureStat(((f-1)*K)+k,9) = norm(DST.parentRealCorr(:));
            dataScrambledTextureStat(((f-1)*K)+k,10) = norm(DST.varianceHPR);
            
            dataTextureStatVec(((f-1)*K)+k, :) = [DT.pixelStats'; DT.pixelLPStats(:); DT.magMeans(:); DT.cousinMagCorr(:); DT.parentMagCorr(:); DT.cousinRealCorr(:); DT.parentRealCorr(:); DT.varianceHPR];
            dataScrambledTextureStatVec(((f-1)*K)+k, :) = [DST.pixelStats'; DST.pixelLPStats(:); DST.magMeans(:); DST.cousinMagCorr(:); DST.parentMagCorr(:); DST.cousinRealCorr(:); DST.parentRealCorr(:); DST.varianceHPR];
            
            dataImg(((f-1)*K)+k,:,:) = imPatch;
            dataScrambledImg(((f-1)*K)+k,:,:) = imScrambledPatch;
            
            d = 256;
            if k == 1
                rIm = [randi(size(im,1)-(d-1)) randi(size(im,2)-(d-1))];
                imgSample(f,1,:,:) = im(rIm(1):rIm(1)+(d-1), rIm(2):rIm(2)+(d-1));
                rIm2 = [randi(size(imRandPhase,1)-(d-1)) randi(size(imRandPhase,2)-(d-1))];
                imgSample(f,2,:,:) = imRandPhase(rIm2(1):rIm2(1)+(d-1), rIm2(2):rIm2(2)+(d-1));
            end

            %apply transfer functions to coefficients and construct data entry
            co = nan(1, J*paramCount);
            cso = nan(1, J*paramCount);
            for j = 1:J
                transFun = coeffTransferSet{j};
                c = imSpyr(coeffIndex);
                ct(((j-1)*paramCount)+1:(j*paramCount)) = transFun(c);
                cw = ct.*coeffWeights(coeffIndex)';

                cs = imScrambledSpyr(coeffIndex);
                cst(((j-1)*paramCount)+1:(j*paramCount)) = transFun(cs);
                csw = cst.*coeffWeights(coeffIndex)'; 
            end

            dataMatrix((f-1)*K+k,:) = cw;
            dataClassificationGroup(((f-1)*K)+k) = 100+FSET(f);
            dataScrambledMatrix((f-1)*K+k,:) = csw;
            dataScrambledClassificationGroup(((f-1)*K)+k) = 200+FSET(f);
        end
    end
    fprintf('\nImages Parsed\n');
    
    DM = [dataMatrix; dataScrambledMatrix];
    DCG = [dataClassificationGroup; dataScrambledClassificationGroup];
    
    %%
    %run pca on data
    [DMcoeff, ~, ~, ~, DMexplained, DMmu] = pca(DM);
end


%% attempt to construct a lower-d basis of coefficients
coeffData = cell(I,J);
for i = 1:I
    for j = 1:J
        %get coefficients
        sampleCoeff = DMcoeff(((j-1)*paramCount)+(1:paramCount),i);
        sampleCoeffFull = zeros(length(coeffIndex),1);
        sampleCoeffFull(coeffIndex) = sampleCoeff;
        
        %parameters
        SN = [1 N N N N N N/2 N/2 N/2 N/2 N/4];
        SNi = cumsum(SN.^2);
        SNuse = [0 2 2 2 2 1 1 1 1 0];
        
        %coeff data structure
        cd.low = nan(4,DZ*DZ);
        cd.high = nan(4,DZ*DZ); 
        cd.Spyr = sampleCoeffFull;
        
        %rescale and dump in correct spot
        for k = 1:length(SN)-1
            coeffSet = reshape(sampleCoeffFull(SNi(k):SNi(k+1)-1), [SN(k+1) SN(k+1)]);
            if SNuse(k) == 0
                %dont use
            elseif SNuse(k) == 2;
                coeffSetImg = imresize(coeffSet, [DZ DZ], 'bilinear');
                cd.high(4-(k-2),:) = coeffSetImg(:);
            elseif SNuse(k) == 1;
                coeffSetImg = imresize(coeffSet, [DZ DZ], 'bilinear');
                cd.low(4-(k-6),:) = coeffSetImg(:);
            end
        end
        
        %save
        coeffData{i,j} = cd;
    end
end

%% attempt PCA on coeffData
for i = 1:I
    for j = 1:J
        close all;
        CD = [coeffData{i,j}.low; coeffData{i,j}.high];
        [CDcoeff, CDscore, ~, ~, CDexplained, CDmu] = pca(CD);
        
        %reconstruct using PCs
        CDRecon = repmat(CDmu, [size(CD,1), 1]);
        for dr = 1:DR;
            CDRecon = CDRecon + CDscore(:,dr) * CDcoeff(:,dr)';
        end
           
        %interpolate reconstruction to full size
        CDReconFull = [];
        SN = [N/4 N/2 N N];
        SNuse = [0 4 4 0];
        for s = 1:length(SN);
            if SNuse(s) == 0
                cm = zeros(SN(s), SN(s));
                CDReconFull = [cm(:); CDReconFull];
            else
                for x = 1:SNuse(s)
                    cm = imresize(reshape(CDRecon((s-2)*4+x,:), [DZ DZ]), [SN(s) SN(s)], 'bilinear');
                    CDReconFull = [cm(:); CDReconFull];
                end
            end
        end
        
%         %Display low-d reconstruction       
%         figure('position', [378 811 560 420])
%         showSpyr(coeffData{i,j}.Spyr, imPind, 'auto1');
%         
%         figure('position', [1278 811 560 420]);
%         
%         subplot(2,2,1);
%         stem(1:length(explained), CDexplained);
%         
%         subplot(2,2,2);
%         imagesc(reshape(CDcoeff(:,1), [F F]));
%         subplot(2,2,3);
%         imagesc(reshape(CDcoeff(:,2), [F F]));
%         subplot(2,2,4);
%         imagesc(reshape(CDcoeff(:,3), [F F]));
%         colormap(gray);
%         
%         figure('position', [378 211 560 420])
%         showSpyr(CDReconFull, imPind, 'auto1');
%         
%         pause;

        %save
        coeffData{i,j}.lowRecon = CDRecon(1:4,:);
        coeffData{i,j}.highRecon = CDRecon(5:8,:);
        coeffData{i,j}.SpyrRecon = CDReconFull;
    end
end


%% apply filters to patches and compare texture statistics
close all;

linearCoeffIndex = 1:sum(coeffIndex);
CBFilter(I).coeff = DMcoeff(:,i);
for i = 1:I
    %save computed filters
    CBFilter(i).coeff = DMcoeff(:,i);
    for k = 1:PSK
        CBFilter(i).coeffPermutation{k} = CBFilter(i).coeff(randperm(length(CBFilter(i).coeff)));
    end
    cr = [];
    for j = 1:J
        cr = [cr coeffData{i,j}.SpyrRecon(coeffIndex)];
    end
    
    %save filter reconstructions;
    CBFilter(i).coeffRecon = cr;
end

%construct response matrix
componentResponseMatrix = zeros(size(DM,1),I);
componentReconResponseMatrix = zeros(size(DM,1),I);
componentResponsePermutationMatrix = zeros(size(DM,1),I, PSK);
CBFilterMatrix = zeros(length(CBFilter(1).coeff), I);
CBReconFilterMatrix = zeros(length(CBFilter(1).coeffRecon), I);
CBFilterPermutationMatrix = zeros(length(CBFilter(1).coeffPermutation{1}), I, PSK);

for i = 1:I
    CBFilterMatrix(:,i) = CBFilter(i).coeff;
    CBReconFilterMatrix(:,i) = CBFilter(i).coeffRecon;
    
    for k = 1:PSK
        CBFilterPermutationMatrix(:,i,k) = CBFilter(i).coeffPermutation{k};
    end
end
componentResponseMatrix(:,:) = DM * CBFilterMatrix;
componentReconResponseMatrix(:,:) = DM * CBReconFilterMatrix;

%preform response statistics
componentResponseStatistics = [mean(componentResponseMatrix, 1);...
                               std(componentResponseMatrix, 1)];
                

for k = 1:PSK
    componentResponsePermutationMatrix(:,:, k) = DM * squeeze(CBFilterPermutationMatrix(:,:,k));
end

CRresponse = componentResponseMatrix;
CRresponse = CRresponse - repmat(min(CRresponse), [size(CRresponse,1), 1]);
CRresponse = CRresponse ./ repmat(max(CRresponse), [size(CRresponse,1), 1]);
CRresponseDKL = zeros(I);

CRRPresponseMean = zeros(2, I, PSK);
CRRPresponseSTD = zeros(2, I, PSK);
CRRPresponseDKL = zeros(I, PSK);

lx = linspace(0,1,25);
for i = 1:I
    p = histc(CRresponse(DCG<200,i), lx);
    p = p / sum(p);
    q = histc(CRresponse(DCG>200,i), lx);
    q = q / sum(q);
    CRresponseDKL(i) = sum(p.*log(p./(q+1e-10)+1e-10));
end
for k = 1:PSK
    CRRPresponse = squeeze(componentResponsePermutationMatrix(:,:,k));
    CRRPresponse = CRRPresponse - repmat(min(CRRPresponse), [size(CRRPresponse,1), 1]);
    CRRPresponse = CRRPresponse ./ repmat(max(CRRPresponse), [size(CRRPresponse,1), 1]);
    CRRPresponseMean(1,:,k) = mean(CRRPresponse(DCG<200,:), 1);
    CRRPresponseMean(2,:,k) = mean(CRRPresponse(DCG>=200,:), 1);
    CRRPresponseSTD(1,:,k) = std(CRRPresponse(DCG<200,:), [], 1);
    CRRPresponseSTD(2,:,k) = std(CRRPresponse(DCG>=200,:), [], 1);
    
    for i = 1:I
        p = histc(CRRPresponse(DCG<200,i), lx);
        p = p / sum(p);
        q = histc(CRRPresponse(DCG>200,i), lx);
        q = q / sum(q);
        CRRPresponseDKL(i, k) = sum(p.*log(p./(q+1e-10)+1e-10));
    end
end

I2 = 1:16;
figure;
subplot(3,1,1);
hold on;
plot(I2, mean(CRresponse(DCG<200,I2)), 'kx');
plot(I2, mean(CRresponse(DCG>=200,I2)), 'rx');
errorbar(I2, mean(squeeze(CRRPresponseMean(1,I2,:)), 2), std(squeeze(CRRPresponseMean(1,I2,:)), [], 2), 'ko');
errorbar(I2, mean(squeeze(CRRPresponseMean(2,I2,:)), 2), std(squeeze(CRRPresponseMean(2,I2,:)), [], 2), 'ro');
ylabel('Response Mean');

subplot(3,1,2);
hold on;
plot(I2, std(CRresponse(DCG<200,I2)), 'kx');
plot(I2, std(CRresponse(DCG>=200,I2)), 'rx');
errorbar(I2, mean(squeeze(CRRPresponseSTD(1,I2,:)), 2), std(squeeze(CRRPresponseSTD(1,I2,:)), [], 2), 'ko');
errorbar(I2, mean(squeeze(CRRPresponseSTD(2,I2,:)), 2), std(squeeze(CRRPresponseSTD(2,I2,:)), [], 2), 'ro');
ylabel('Response STD');

subplot(3,1,3);
hold on;
plot(I2, CRresponseDKL(I2), 'kx')
errorbar(I2, mean(CRRPresponseDKL(I2,:), 2), std(CRRPresponseDKL(I2,:), [], 2), 'ko');
ylabel('KL Divergence');
xlabel('Component Number');

figure('position', [200 50 1600 500]);

for i = 1:8
    subplot(2,8,i);
    y1 = histc(CRresponse(DCG<200,i), lx);
    y2 = histc(CRresponse(DCG>=200,i), lx);
    y1 = y1 / sum(y1);
    y2 = y2 / sum(y2);
    hold on;
    plot(lx, y1, 'k-');
    plot(lx, y2, 'r-');
    set(gca, 'xlim', [0 1], 'ylim', [0 1.75/sqrt(length(lx))]);
    if i == 1
        ylabel('Probability')
        legend({'Texture', 'Noise'})
    else
        set(gca, 'yticklabel', {})
    end
    title(['Component #' num2str(i)]);    
    
    subplot(2,8,8+i);
    y1 = histc(CRRPresponse(DCG<200,i), lx);
    y2 = histc(CRRPresponse(DCG>=200,i), lx);
    y1 = y1 / sum(y1);
    y2 = y2 / sum(y2);
    hold on;
    plot(lx, y1, 'k-');
    plot(lx, y2, 'r-');
    set(gca, 'xlim', [0 1], 'ylim', [0 .5]);
    if i == 1
        ylabel('Frequency')
        xlabel('Normalized Response')
    else
        set(gca, 'yticklabel', {})
    end
    title(['Randomized #' num2str(i)]);  
end

a = 1:8;
[x, y] = meshgrid(a,a);
x = x(tril(true(size(x)),-1));
y = y(tril(true(size(y)), -1));
ps = [x(:) y(:)];

DMTexStat = [dataTextureStat(:, [1:2 5:10]); dataScrambledTextureStat(:, [1:2 5:10])];
DMCRTS = [componentResponseMatrix(:,1:8) componentResponseMatrix(:,1:8).^2 DMTexStat(:,1:8)];
DMCRTSps = [componentResponseMatrix(:,ps(:,1)).*componentResponseMatrix(:,ps(:,2)) DMTexStat(:,1:8)];
DMCRPermTS = [componentResponsePermutationMatrix(:,1:8) componentResponsePermutationMatrix(:,1:8).^2 DMTexStat(:,1:8)];
DMCRPermTSps = [componentResponsePermutationMatrix(:,ps(:,1)).*componentResponsePermutationMatrix(:,ps(:,2)) DMTexStat(:,1:8)];

DMCRTS = DMCRTS - repmat(mean(DMCRTS), [size(DM,1) 1]);
DMCRTS = DMCRTS ./ repmat(std(DMCRTS), [size(DM,1) 1]);
DMCRTSps = DMCRTSps - repmat(mean(DMCRTSps), [size(DM,1) 1]);
DMCRTSps = DMCRTSps ./ repmat(std(DMCRTSps), [size(DM,1) 1]);
DMCRPermTS = DMCRPermTS - repmat(mean(DMCRPermTS), [size(DM,1) 1]);
DMCRPermTS = DMCRPermTS ./ repmat(std(DMCRPermTS), [size(DM,1) 1]);
DMCRPermTSps = DMCRPermTSps - repmat(mean(DMCRPermTSps), [size(DM,1) 1]);
DMCRPermTSps = DMCRPermTSps ./ repmat(std(DMCRPermTSps), [size(DM,1) 1]);

DMcov = cov(DMCRTS);
DMpscov = cov(DMCRTSps);
DMPermcov = cov(DMCRPermTS);
DMPermpscov = cov(DMCRPermTSps);

DMpsticks = {};
for i = 1:length(ps)
    DMpsticks = {DMpsticks{:} ['c' num2str(ps(i,1)) 'c' num2str(ps(i,2))]};
end

B = CLASSIFICATION_VALIDATION_BOOTSTRAP;
P = CLASSIFICATION_VALIDATION_SIZE;

%% fit texture statistics
if FLAG_DO_TEXTURESTATS

    CK = {'distance', 'seuclidean', 'depth', 8, 'cutoff', 1.75};

    h = figure('position', [300 100 1600 1100]);
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

    subplot(2,3,1);
    imagesc(1:8, 1:8, DMcov(17:end, 17:end));
    ylabel('Normed Texture Statistic');
    xlabel('Normed Texture Statistic');
    colorbar;
    title('Texture Metric');

    subplot(2,3,2);
    DMcovMat = DMcov(1:16, 17:end);
    DMcovMatLabel = {'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c1c1', 'c2c2', 'c3c3', 'c4c4', 'c5c5','c6c6', 'c7c7', 'c8c8'};
    T = clusterdata(DMcovMat, CK{:});
    [~, Ti] = sort(T);
    imagesc(1:8, 1:16, DMcovMat(:,:));
    ylabel('Component Response');
    xlabel('Normed Texture Statistic');
    set(gca, 'yTick', 1:16, 'yTickLabel', DMcovMatLabel);
    colorbar;
    title('Learned Pyramid Components');

    subplot(2,3,3);
    DMpscovMat =  DMpscov(1:length(ps), length(ps)+1:end);
    DMpscovMatLabel = DMpsticks;
    T = clusterdata(DMpscovMat, CK{:});
    [~, Ti] = sort(T);
    imagesc(1:8, 1:length(ps), DMpscovMat(Ti,:))
    ylabel('Cross Component Response');
    xlabel('Normed Texture Statistic');
    set(gca, 'yTick', 1:length(DMpsticks), 'yTickLabel', DMpscovMatLabel(Ti));
    colorbar;
    title('Learned Pyramid Components');

    subplot(2,3,5);
    DMPermcovMat =  DMPermcov(1:16, 17:end);
    DMPermcovMatLabel = DMcovMatLabel;
    T = clusterdata(DMPermcovMat, CK{:});
    [~, Ti] = sort(T);
    imagesc(1:8, 1:16, DMPermcovMat(:,:));
    ylabel('Component Permutation Response');
    xlabel('Normed Texture Statistic');
    set(gca, 'yTick', 1:16, 'yTickLabel', DMPermcovMatLabel);
    colorbar;
    title('Scrambled Pyramid Components');

    subplot(2,3,6);
    DMPermpscovMat =  DMPermpscov(1:length(ps), length(ps)+1:end);
    DMPermpscovMatLabel = DMpsticks;
    T = clusterdata(DMPermpscovMat, CK{:});
    [~, Ti] = sort(T);
    imagesc(1:8, 1:length(ps), DMPermpscovMat(Ti,:))
    ylabel('Cross Component Permutation Response');
    xlabel('Normed Texture Statistic');
    set(gca, 'yTick', 1:length(DMpsticks), 'yTickLabel', DMPermpscovMatLabel(Ti));
    colorbar;
    title('Scrambled Pyramid Components');

    colormap(flipud(othercolor('RdBu9')));

end

%% texture statistic prediction
Iset = (1:2:21);

%cdatacross = cdata;
DMtsv = [dataTextureStatVec; dataScrambledTextureStatVec];

%normalize texture statistics matrix
DMtsv = DMtsv - repmat(mean(DMtsv,1), [size(DMtsv,1) 1]);
bi = std(DMtsv) == 0;
DMtsv = DMtsv(:,~bi);
DMtsv = DMtsv ./ repmat(std(DMtsv), [size(DMtsv,1) 1]);

cdata = DM * CBFilterMatrix;
crdata = DM * CBReconFilterMatrix;
crrpdata = DM * CBFilterPermutationMatrix(:,:,randi(PSK));
cdatacg = round(DCG/100);

if FLAG_DO_TEXTURESTATS
    statData = [];
    statData.s{B} = zeros(1,size(cdata,1));
    statData.regCrossMat{B} = {};
    statData.regLsqMat{B} = {};
    statDataSet(length(Iset)) = statData;

    fprintf('Predicting Texture Statistics:');
    for i = Iset
        %expand cdata to include square/cross terms
        a = 1:i;
        [x, y] = meshgrid(a,a);
        x = x(tril(true(size(x)), 0));
        y = y(tril(true(size(y)), 0));
        ps = [x(:) y(:)];
        cdataCross = [cdata(:,a) cdata(:, ps(:,1)) .* cdata(:, ps(:,2))];
        cdataLsq = [cdata(:,a)];

        fprintf(' %i', i);
        for b = 1:B
            s = false(1, size(cdata,1));
            s(1:round(length(s)*P)) = true;
            s = s(randperm(length(s)));

            %construct solution matrix
            regCrossMat = zeros(size(cdataCross, 2), size(DMtsv, 2));
            regLsqMat = zeros(size(cdataLsq, 2), size(DMtsv, 2));

            %for every variable, solve regression against data matrix
            for k = 1:size(regCrossMat, 2)
                regCrossMat(:,k) = lsqlin(cdataCross(~s,:), DMtsv(~s,k));
                regLsqMat(:,k) = lsqlin(cdataLsq(~s,:), DMtsv(~s,k));
            end

            %apply solution to holdout data
            hCrossResiduals = cdataCross(s,:)*regCrossMat - DMtsv(s,:);
            hLsqResiduals = cdataLsq(s,:)*regLsqMat - DMtsv(s,:);
            hMeanModelResiduals = repmat(mean(DMtsv(~s,:),1), [sum(s) 1]) - DMtsv(s,:);

            %get mean vector length of training data;
            statLength = mean(sqrt(sum(DMtsv(~s,:).^2, 2)));

            %save error
            statPredictionCrossError(i,b) = mean(sqrt(sum(hCrossResiduals.^2, 2))) / statLength;
            statPredictionLsqError(i,b) = mean(sqrt(sum(hLsqResiduals.^2, 2))) / statLength;
            statMeanModelError(i,b) = mean(sqrt(sum(hMeanModelResiduals.^2, 2))) / statLength;

            %save data for offline analysis
            statDataSet(i).s{b} = s;
            statDataSet(i).regCrossMat{b} = regCrossMat;
            statDataSet(i).regLsqMat{b} = regLsqMat;
        end
    end
    fprintf('.\n');

    sds = statDataSet(Iset);
    statPredictionCrossError = zeros(length(Iset),B);
    statPredictionLsqError = zeros(length(Iset),B);
    statMeanModelError = zeros(length(Iset),B);

    statPredictionCrossError2 = zeros(length(Iset),B);
    statPredictionLsqError2 = zeros(length(Iset),B);
    statMeanModelError2 = zeros(length(Iset),B);

    for i = 1:length(sds);

        a = 1:Iset(i);
        [x, y] = meshgrid(a,a);
        x = x(tril(true(size(x)), 0));
        y = y(tril(true(size(y)), 0));
        ps = [x(:) y(:)];
        cdataCross = [cdata(:,a) cdata(:, ps(:,1)) .* cdata(:, ps(:,2))];
        cdataLsq = [cdata(:,a)];

        hCrossResiduals = zeros(sum(sds(i).s{1}), size(sds(i).regCrossMat{1},2));
        for b = 1:B
            s = sds(i).s{b};
            hCrossResiduals = hCrossResiduals + abs(cdataCross(s,:)*sds(i).regCrossMat{1} - DMtsv(s,:));
        end
        hCrossResiduals = hCrossResiduals / B;
    %     subplot(1, length(Iset), i)
    %     hist(sum(hCrossResiduals,1)/size(hCrossResiduals,1));

        if i == 8
            statSet = sum(hCrossResiduals,1)/size(hCrossResiduals,1) < .5;
        end
    end

    %contruct vector for all stimuli to hold statistic prediction error
    %...to hold error bootstrap trial counts
    patchPredictionError = zeros(length(sds), length(DCG), B);
    patchPredictionCount = zeros(length(sds), length(DCG), B);

    for i = 1:length(sds);

        a = 1:Iset(i);
        [x, y] = meshgrid(a,a);
        x = x(tril(true(size(x)), 0));
        y = y(tril(true(size(y)), 0));
        ps = [x(:) y(:)];
        cdataCross = [cdata(:,a) cdata(:, ps(:,1)) .* cdata(:, ps(:,2))];
        cdataLsq = cdata(:,a);

        for b = 1:B
            s = sds(i).s{b};

            %compute 2norm of all stats
            statLength = mean(sqrt(sum(DMtsv(~s,:).^2, 2)));

            hCrossResiduals = cdataCross(s,:)*sds(i).regCrossMat{b}(:,:) - DMtsv(s,:);
            hLsqResiduals = cdataLsq(s,:)*sds(i).regLsqMat{b}(:,:) - DMtsv(s,:);
            hMeanModelResiduals = repmat(mean(DMtsv(~s,:),1), [sum(s) 1]) - DMtsv(s,:);

            statPredictionCrossError(i,b) = mean(sqrt(sum(hCrossResiduals.^2, 2)) / statLength);
            statPredictionLsqError(i,b) = mean(sqrt(sum(hLsqResiduals.^2, 2)) / statLength);
            statMeanModelError(i,b) = mean(sqrt(sum(hMeanModelResiduals.^2, 2)) / statLength);

            %add prediction errors to vectors
            patchPredictionError(i,s,b) = sqrt(sum(hCrossResiduals.^2, 2)) / statLength;
            patchPredictionCount(i,s,b) = patchPredictionCount(i,s) + 1;

            %compute 2norm of select stats that are reduced the most
            statLength2 = mean(sqrt(sum(DMtsv(~s,statSet).^2, 2)));

            hCrossResiduals2 = cdataCross(s,:)*sds(i).regCrossMat{b}(:,statSet) - DMtsv(s,statSet);
            hLsqResiduals2 = cdataLsq(s,:)*sds(i).regLsqMat{b}(:,statSet) - DMtsv(s,statSet);
            hMeanModelResiduals2 = repmat(mean(DMtsv(~s,statSet),1), [sum(s) 1]) - DMtsv(s,statSet);

            statPredictionCrossError2(i,b) = mean(sqrt(sum(hCrossResiduals2.^2, 2)) / statLength2);
            statPredictionLsqError2(i,b) = mean(sqrt(sum(hLsqResiduals2.^2, 2)) / statLength2);
            statMeanModelError2(i,b) = mean(sqrt(sum(hMeanModelResiduals2.^2, 2)) / statLength2);

        end
    end

    %% plot texture statistic prediction errors
    figure; hold on;
    errorbar(Iset, mean(statMeanModelError(:,:),2), std(statMeanModelError(:,:), [], 2), 'ko-', 'lineWidth', 2, 'markerSize', 8);
    errorbar(Iset, mean(statPredictionLsqError(:,:),2), std(statPredictionLsqError(:,:), [], 2), 'o-', 'color', [0    0.4980         0], 'lineWidth', 2, 'markerSize', 8);
    errorbar(Iset, mean(statPredictionCrossError(:,:),2), std(statPredictionCrossError(:,:), [], 2), 'o-', 'color', [0.4784    0.0627    0.8941], 'lineWidth', 2, 'markerSize', 8);
    axis([0 22 .5 1.1]);
    set(gca, 'xtick', 1:4:max(Iset), 'ytick', .5:.1:1.1); 
    ylabel('Normalized Texture Statistics Error');
    xlabel('Components');
    legend({'Mean Model', 'Linear Model', 'Quadratic Model'});
    title('All Statistics');

    figure; hold on;
    errorbar(Iset, mean(statMeanModelError2(:,:),2), std(statMeanModelError2(:,:), [], 2), 'k.-');
    errorbar(Iset, mean(statPredictionLsqError2(:,:),2), std(statPredictionLsqError2(:,:), [], 2), '.-', 'color', [0    0.4980         0]);
    errorbar(Iset, mean(statPredictionCrossError2(:,:),2), std(statPredictionCrossError2(:,:), [], 2), '.-', 'color', [0.4784    0.0627    0.8941]);
    ylabel('Normalized Texture Statistics Error');
    xlabel('Components');
    legend({'Mean Model', 'Linear', 'Quadratic'});
    title('Best-Predicted Statistics');

end

%% preform classification  
componentError_l = nan(1,I); 
componentError_l2 = nan(1,I);
componentError_q = nan(1,I);
componentReconError_l = nan(1,I);
componentReconError_q = nan(1,I);
componentRandError_q = nan(1,I);

if FLAG_DO_TEXTURESTATS
    patchClassifyPredictionErrors = [];
    patchClassifyPredictionErrors(length(Iset)).hit = [];
    patchClassifyPredictionErrors(length(Iset)).falsePositive = [];
    patchClassifyPredictionErrors(length(Iset)).miss = [];
    patchClassifyPredictionErrors(length(Iset)).reject = [];
end
for i = 1:I;
    
    bootstrap_l = zeros(1,B);
    bootstrap_l2 = zeros(1,B);
    bootstrap_q = zeros(1,B);
    bootstrapR_l = zeros(1,B);
    bootstrapR_q = zeros(1,B);
    bootstrapRand_q = zeros(1,B);
    
    for b = 1:B
        
        if FLAG_DO_TEXTURESTATS
            if(ismember(i, Iset))
                s = sds(Iset == i).s{b};
            else
                s = false(1, size(cdata,1));
                s(1:round(length(s)*P)) = true;
                s = s(randperm(length(s)));
            end
        else
            s = false(1, size(cdata,1));
            s(1:round(length(s)*P)) = true;
            s = s(randperm(length(s)));
        end

        classifySample = cdata(s, 1:i);
        classifySample2 = [cdata(s, 1:i) cdata(s, 1:i).^2];
        classifyReconSample = crdata(s, 1:i);
        classifyRandSample = crrpdata(s, 1:i);
        classifySampleGroup = cdatacg(s);
        classifyTraining = cdata(~s,1:i);
        classifyTraining2 = [cdata(~s,1:i) cdata(~s,1:i).^2];
        classifyReconTraining = crdata(~s,1:i);
        classifyRandTraining = crrpdata(~s,1:i);
        classifyTrainingGroup = cdatacg(~s);

        [class_l, err_l] = classify(classifySample, classifyTraining, classifyTrainingGroup, 'linear');
        [class_l2, err_l2] = classify(classifySample2, classifyTraining2, classifyTrainingGroup, 'linear');
        [class_q, err_q] = classify(classifySample, classifyTraining, classifyTrainingGroup, 'quadratic');
        bootstrap_l(b) = 1-sum(class_l == classifySampleGroup)/length(class_l);
        bootstrap_l2(b) = 1-sum(class_l2 == classifySampleGroup)/length(class_l2);
        bootstrap_q(b) = 1-sum(class_q == classifySampleGroup)/length(class_q);
        
        [classR_l, errR_l] = classify(classifyReconSample, classifyReconTraining, classifyTrainingGroup, 'linear');
        [classR_q, errR_q] = classify(classifyReconSample, classifyReconTraining, classifyTrainingGroup, 'quadratic');
        bootstrapR_l(b) = 1-sum(classR_l == classifySampleGroup)/length(classR_l);
        bootstrapR_q(b) = 1-sum(classR_q == classifySampleGroup)/length(classR_q);
        
        [classRand_q, errRand_q] = classify(classifyRandSample, classifyRandTraining, classifyTrainingGroup, 'quadratic');
        bootstrapRand_q(b) = 1-sum(classRand_q == classifySampleGroup)/length(classRand_q);
       
        if FLAG_DO_TEXTURESTATS
            if ismember(i, Iset)
                %for quadratic classifier, check predictions
                e = patchPredictionError(Iset == i,s, b);

                patchClassifyPredictionErrors(Iset == i).hit = [patchClassifyPredictionErrors(Iset == i).hit e(classRand_q == 1 & classifySampleGroup == 1)];
                patchClassifyPredictionErrors(Iset == i).falsePositive = [patchClassifyPredictionErrors(Iset == i).falsePositive e(classRand_q == 2 & classifySampleGroup == 1)];
                patchClassifyPredictionErrors(Iset == i).miss = [patchClassifyPredictionErrors(Iset == i).miss e(classRand_q == 1 & classifySampleGroup == 2)];
                patchClassifyPredictionErrors(Iset == i).reject = [patchClassifyPredictionErrors(Iset == i).reject e(classRand_q == 2 & classifySampleGroup == 2)];
            end
        end
    end
    
    componentError_l(i) = mean(bootstrap_l);
    componentError_l2(i) = mean(bootstrap_l2);
    componentError_q(i) = mean(bootstrap_q);
    componentReconError_l(i) = mean(bootstrapR_l);
    componentReconError_q(i) = mean(bootstrapR_q);
    componentRandError_q(i) = mean(bootstrapRand_q);
end

%% look at texture statistic error histograms of classification groups

if FLAG_DO_TEXTURESTATS
    xset = linspace(0,2,30);
    for i = Iset
        clf; hold on;

        errorHit = histc(patchClassifyPredictionErrors(Iset == i).hit, xset);
        errorFalsePositive = histc(patchClassifyPredictionErrors(Iset == i).falsePositive, xset);
        errorMiss = histc(patchClassifyPredictionErrors(Iset == i).miss, xset);
        errorReject = histc(patchClassifyPredictionErrors(Iset == i).reject, xset);


        plot(xset, errorHit / sum(errorHit), 'r-');
        plot(xset, errorFalsePositive / sum(errorFalsePositive), 'm-');
        plot(xset, errorMiss / sum(errorMiss), 'c-');
        plot(xset, errorReject / sum(errorReject), 'b-');

        title(num2str(i));
        ylabel('Frequency')
        xlabel('Texture statistics error');
        set(gca, 'ytick', [0:.1:.4], 'ylim', [0 .4], 'xtick', 0:1:2, 'xlim', [0 2]);
        pause; 
    end
end

%%
close all;

%percentage of explained variance
pev = DMexplained/(sum(DMexplained));

n1 = 0;
for j = 1:J
    for ii = 1:16
        sampleCoeff = zeros(length(coeffIndex),1);
        sampleCoeff(coeffIndex) = DMcoeff(((j-1)*paramCount) + (1:paramCount),ii);
        imRecon = putSpyr(sampleCoeff, imPind);
        if max(abs(imRecon(:))) > n1;
            n1 = max(abs(imRecon(:)));
        end
    end
end

%for every coefficient transfer function
for j = 1:J
    %texture filters
    figure('position', [620+(j-1)*40 580-(j-1)*40 560 400]);
    hold on;
    for ii = 1:16
        sampleCoeff = zeros(length(coeffIndex),1);
        sampleCoeff(coeffIndex) = DMcoeff(((j-1)*paramCount) + (1:paramCount),ii);
        imRecon = putSpyr(sampleCoeff, imPind);
        
        subplot(4,4,ii);
        colormap(gray(256));
        image(127+128*(imRecon/n1))
        axis equal tight off;
        
        if ii == 1
            title(['Tf: ' char(coeffTransferSet{j})])
        else
            title(['Filter #' num2str(ii)]);
        end
    end
end

%Show variance captured by each component
figure('position', [1220 580 560 400]);
hold on;
stem(1:min(length(pev),64), pev(1:min(length(pev),64)));
ylabel('Percent explained variance');
xlabel('Component number');

%show image and phase scramble;
for f = 1:F
    figure('position', [1220+(f-1)*40 120+(f-1)*40 560 400]);
    subplot(4,4,[1 2 5 6]);
    colormap('gray');
    imagesc(squeeze(imgSample(f,1,:,:)));
    axis equal off tight;
    title('Image');

    subplot(4,4,[3 4 7 8]);
    colormap('gray');
    imagesc(squeeze(imgSample(f,2,:,:)));
    axis equal off tight;
    title('Other Image');
    
    iSub = [9 10 13 14];
    oiSub = [11 12 15 16];
    for i = 1:2
        subplot(4,4,iSub(i));
        colormap('gray');
        imagesc(squeeze(imgSampleSmall(i,f,1,:,:)));
        axis equal off tight;

        subplot(4,4,oiSub(i));
        colormap('gray');
        imagesc(squeeze(imgSampleSmall(i,f,2,:,:)));
        axis equal off tight;
    end
    for i = 1:2
        subplot(4,4,iSub(i+2));
        colormap('gray');
        imagesc(squeeze(imgSampleWeighted(i,f,1,:,:)));
        axis equal off tight;

        subplot(4,4,oiSub(i+2));
        colormap('gray');
        imagesc(squeeze(imgSampleWeighted(i,f,2,:,:)));
        axis equal off tight;
    end
end

for i = 8:-1:1
    for j = 1:J
        %get top coefficients and plot them
        sampleCoeff = zeros(length(coeffIndex),i);
        sampleCoeff(coeffIndex) = DMcoeff(((j-1)*paramCount)+(1:paramCount),i);
        f1 = figure('position', [20+20*(i-1)+((j-1)*20) 580-20*(i-1) 560 400]);
        showSpyr(sampleCoeff, imPind, 'auto1');
        title(sprintf('Component: #%i, transfer: %s', i, char(coeffTransferSet{j})));
    end
end

for i = 8:-1:1
    for j = 1:J
        %get top coefficients and plot them
        sampleCoeff = coeffData{i,j}.SpyrRecon;
        f1 = figure('position', [20+20*(i-1)+((j-1)*20) 280-20*(i-1) 560 400]);
        showSpyr(sampleCoeff, imPind, 'auto1');
        title(sprintf('Reconstructed Component: #%i, transfer: %s', i, char(coeffTransferSet{j})));
    end
end

figure('position', [620 60 560 400]);
cmap = hsv(F);

sSet = randperm(size(DCG,1));
DMs = DM(sSet(1:J*F*K),:);
DCGs = DCG(sSet(1:J*F*K));
for i = 1:4
    fi = 2*(i-1)+1;
    subplot(2,2,i);
    hold on;
    for f = 1:F
        plot(componentResponseMatrix(DCGs == 100+f, fi), componentResponseMatrix(DCGs == 100+f, fi+1), 'kx', 'color', cmap(f,:));
        plot(componentResponseMatrix(DCGs == 200+f, fi), componentResponseMatrix(DCGs == 200+f, fi+1), 'ko', 'color', cmap(f,:));
    end
    xlabel(sprintf('Texture Filter #%i Response', fi));
    ylabel(sprintf('Texture Filter #%i Response', fi+1));
end

figure('position', [640 40 560 400]);
sSet = randperm(size(DCG,1));
DMs = DM(sSet(1:J*F*K),:);
DCGs = DCG(sSet(1:J*F*K));
for i = 1:4
    fi = 2*(i-1)+1;
    subplot(2,2,i);
    hold on;
    for f = 1:F
        plot(componentResponseMatrix(DCGs == 100+f, fi), componentResponseMatrix(DCGs == 100+f, fi+1), 'b.');
        plot(componentResponseMatrix(DCGs == 200+f, fi), componentResponseMatrix(DCGs == 200+f, fi+1), 'r.');
    end
    xlabel(sprintf('Texture Filter #%i Response', fi));
    ylabel(sprintf('Texture Filter #%i Response', fi+1));
end

figure('position', [20 80 560 400])
hold on;
set(gca, 'xtick', 0:16:128, 'yTick', .4:.1:1, 'yLim', [.4 1]);

plot(1:I, 1-componentError_l, '.-', 'color', [0    0.4980         0], 'lineWidth', 2, 'MarkerSize', 20);
plot(1:I, 1-componentError_l2, '.-', 'color', [0.8941    0.4980         0.0627], 'lineWidth', 2, 'MarkerSize', 20);
plot(1:I, 1-componentError_q, '.-', 'color', [0.4784    0.0627    0.8941], 'lineWidth', 2, 'MarkerSize', 20);

plot([0 I],[0.5 0.5], 'k:');
legend({'Linear', 'Linear with squares', 'Quadratic'});

xlabel('PCA components used');
ylabel('Classification accuracy');

figure('position', [20 80 560 400])
hold on;
set(gca, 'xtick', 0:16:128, 'yTick', .4:.1:1, 'yLim', [.4 1]);

plot(1:I, 1-componentError_l, '.-', 'color', [0    0.4980         0], 'lineWidth', 2, 'MarkerSize', 20);
plot(1:I, 1-componentReconError_l, 'o-', 'color', [0    0.4980         0], 'lineWidth', 2, 'MarkerSize', 8);
plot(1:I, 1-componentError_q, '.-', 'color', [0.4784    0.0627    0.8941], 'lineWidth', 2, 'MarkerSize', 20);
plot(1:I, 1-componentReconError_q, 'o-', 'color', [0.4784    0.0627    0.8941], 'lineWidth', 2, 'MarkerSize', 8);
plot(1:I, 1-componentRandError_q, 'x-', 'color', [0.4784    0.0627    0.8941], 'lineWidth', 2, 'MarkerSize', 10);

plot([0 I],[0.5 0.5], 'k:');
legend({'Linear', 'Linear (low rank)', 'Quadratic', 'Quadratic (low rank)', 'Quadratic (scrambled)'});

xlabel('Number of PCA components');
ylabel('Classification Accuracy');

s = '';
for j = 1:J
    s = [s ' ' char(coeffTransferSet{j})];
end
title(sprintf('Transfer Set: %s', s));

%% save data

data.input.imgSample = imgSample;
data.input.imgSampleSmall = imgSampleSmall;
data.input.dataMatrix = DM;
data.input.dataClassificationGroup = DCG;
data.input.dataTextureStatistics = [dataTextureStatVec; dataScrambledTextureStatVec];
data.input.F = F;
data.input.dataImg = dataImg;
data.input.dataScrambledImg = dataScrambledImg;

data.f.filter = CBFilter;
data.f.coeffWeights = coeffWeights;
data.f.coeffIndex = coeffIndex;
data.f.coeffTransferSet = coeffTransferSet;
data.f.componentResponseMatrix = componentResponseMatrix;
data.f.compomnentResponseStasitics = componentResponseStatistics;
data.f.componentError_l = componentError_l;
data.f.componentError_q = componentError_q;

data.spyr.filts = filts;
data.spyr.imPind = imPind;

save(sprintf('bundle_%i_%i_%s_%i', N, M, regexprep(num2str(FSET), '\s+', '-'), J), 'data');

% set aside data model, ie weights, coeffs, and nonlinearity
dataModel.weights = coeffWeights;
dataModel.components = zeros(length(coeffWeights),length(CBFilter));
for i = 1:length(CBFilter)
    dataModel.components(coeffIndex,i) = CBFilter(i).coeff;
end

if strcmp('@(x)(x.^2+.001).^(1/2)', func2str(coeffTransferSet{:}))
    dataModel.transferFunction = 'softAbs';
else
    dataModel.transferFunction = 'unknown';
end
save(sprintf('bundle_%i_%i_%s_%i_model', N, M, regexprep(num2str(FSET), '\s+', '-'), J), 'dataModel');

fprintf('Saved %s\n', sprintf('bundle_%i_%i_%s_%i_model', N, M, regexprep(num2str(FSET), '\s+', '-'), J));