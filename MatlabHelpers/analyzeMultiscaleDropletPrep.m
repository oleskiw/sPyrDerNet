close all;
clear all;
clc;

addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\ExpoMatlab');
addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\ExpoMatlab\plot_conventions');
addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\ExpoMatlab\Helper functions');
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools');
addpath('C:\Users\oleskiw\Google Drive\wrx\Program\Toolbox\matlabPyrTools\MEX')

FORCE_PROCESS = true;
VERBOSE = true;

dataDir = 'C:\Users\Oleskiw\Google Drive\wrx\Program\V2DropletExperiment\Data\';
parsedDir = [dataDir 'parsed_6o_1p6\'];
figDir = [parsedDir 'figs\'];
mkdir(figDir);

load expData
presentationDuration = 0.200;

%E = [991];
E = [9];
%E = [18 20 21 22];
%E = [3 5 14 15 16 17 18 19 20 21 22];
%E = 1:length(expData);

MULTI{991}.animalName = 'm685';
MULTI{991}.unitName = 'p1#6';
MULTI{991}.setName = 'R_ExpA_sf1000_d20_n256';

MULTI{992}.animalName = 'm685';
MULTI{992}.unitName = 'p1#7';
MULTI{992}.setName = 'R_ExpA_sf1000_d20_n256';

for e = E
    PARSE_MULTI = false;
    if isfield(MULTI{e}, 'animalName')
        PARSE_MULTI = true;
        animalName = MULTI{e}.animalName;
        unitName = MULTI{e}.unitName;
        setName = MULTI{e}.setName;
        
        setSuffix = '';
        clear 'rec';
        
        %load multiunit record
        disp('loading multi rec...');
        load([dataDir animalName filesep animalName unitName filesep 'rec-' animalName unitName '.mat'])
        
        %single unit parameters
        rec.info.Pyr.resolution = 256;
        rec.info.Pyr.scales = 5;
        rec.info.Pyr.orientations = 6;
        rec.info.Pyr.padding = 64;

        rec.info.Pyr.mask.highres = false;
        rec.info.Pyr.mask.lowres = true;
        rec.info.Pyr.mask.highscale = false;
        rec.info.Pyr.mask.lowscale = true;

        rec.info.Ds.scale = 3;
        rec.info.Ds.radius = rec.info.Pyr.resolution/3;
        
    else
        animalName = expData{e,2};
        unitName = expData{e,3};
        setName = expData{e,4};

        setSuffix = '';
        clear 'rec';

        %single unit parameters
        rec.info.Pyr.resolution = 128;
        rec.info.Pyr.scales = 5;
        rec.info.Pyr.orientations = 6;
        rec.info.Pyr.padding = 64;

        rec.info.Pyr.mask.highres = false;
        rec.info.Pyr.mask.lowres = true;
        rec.info.Pyr.mask.highscale = false;
        rec.info.Pyr.mask.lowscale = true;

        rec.info.Ds.scale = 1.6;
        rec.info.Ds.radius = rec.info.Pyr.resolution/3;
    end

    %check for parsed datafile
    if ~exist([parsedDir animalName unitName setName '.mat'], 'file') || FORCE_PROCESS

        if ~PARSE_MULTI
            %prepare data files
            fname = [animalName unitName '#*[timFixStimImg' setSuffix '].xml'];
            fdirname = [dataDir animalName '\' animalName unitName];

            %load expo data
            expFiles = dir(fullfile([fdirname '\' fname]));
            disp(['Analysing ' animalName unitName ', ' num2str(length(expFiles)) ' file(s):']);
            for i = 1:length(expFiles)
                disp(['-processing ' expFiles(i).name]);
                if i == 1
                    data = ReadExpoXML([fdirname '\' expFiles(i).name], 0, 0, 1);
                else
                    dNext = ReadExpoXML([fdirname '\' expFiles(i).name], 0, 0, 1);
                    data = MergeExpoData({data, dNext});
                end
            end
        end

        %save naming information
        rec.name.animal = animalName;
        rec.name.unit = unitName;
        rec.name.set = setName;

        %%
        disp('Loading stimulus data...');
        %load stimulus data
        load([dataDir animalName filesep animalName unitName filesep setName '\stimData_' setName]);

        if ~PARSE_MULTI
            %% single unit, read expo
            disp('Parsing expo records...');
            matrixBlockIDs = data.matrix.MatrixBaseID : 1 : data.matrix.MatrixBaseID+data.matrix.NumOfBlocks-1;

            stimPassIDs = data.passes.IDs(data.passes.SlotIDs == 5);
            selectPassIDs = data.passes.IDs(data.passes.SlotIDs == 4);
            chunkStartPassIDs = data.passes.IDs(data.passes.SlotIDs ==3);
            chunkStimPassIDs = intersect(chunkStartPassIDs+2, stimPassIDs);

            % Get spike template channel
            if ~exist('channel','var')
                channel = 0;
            elseif isempty(channel)
                channel = 0;
            end

            %
            rec.info.stimDuration = presentationDuration;
            spkPool = rec.info.stimDuration;
            spkDuration = 2*rec.info.stimDuration;
            spkOffset = 0.060;
            spkPad = 2*rec.info.stimDuration;

            rec.expo.texid = GetEvents(data, selectPassIDs, 'Op Variable', 'texid', 0, 'Destination');
            rec.expo.ori = GetEvents(data, selectPassIDs, 'Op Variable', 'ori', 0, 'Destination');
            rec.expo.phase = GetEvents(data, selectPassIDs, 'Op Variable', 'phase', 0, 'Destination');
            rec.expo.selectTimes = double(data.passes.StartTimes(ismember(data.passes.IDs, selectPassIDs)));
            rec.expo.presentTimes = double(data.passes.StartTimes(ismember(data.passes.IDs, stimPassIDs)));
            rec.expo.presentEndTimes = double(data.passes.EndTimes(ismember(data.passes.IDs, stimPassIDs)));

            rec.expo.spike = GetSpikeTimes(data, 0, stimPassIDs, spkOffset, spkOffset, 0, 'sec');
            rec.expo.spikeTime = spkOffset:.0001:(spkOffset+spkDuration);
            rec.expo.spikeWide = GetSpikeTimes(data, 0, stimPassIDs, -1*spkPad, 1*spkPad, 0, 'sec');
            rec.expo.spikeWideTime = (-1*spkPad):.0001:spkDuration+(1*spkPad);
            rec.expo.spikeChunk = GetSpikeTimes(data, 0, chunkStimPassIDs, -1*spkPad, 1*spkPad, 0, 'sec');
            rec.expo.spikeChunkTime = (-1*spkPad):.0001:spkDuration+(1*spkPad);

            %save stim params
            rec.info.presentTimes = rec.expo.presentTimes;
            rec.info.presentTex = rec.expo.texid;
            rec.info.presentOri = rec.expo.ori;
            rec.info.presentImg = mod(rec.expo.phase + 1,64);
            rec.info.presentImg(rec.info.presentImg == 0) = 64;

            if length(rec.expo.selectTimes) > length(rec.expo.presentTimes)
                for i = 1:length(rec.info.presentTimes)
                    d = rec.info.presentTimes(i) - rec.expo.selectTimes(i);
                    if d > 2
                        disp(['Stimulus selection and presentation discrepency, dropping trial ' num2str(i)]);
                        rec.info.presentTex = [rec.info.presentTex(1:i-1); rec.info.presentTex(i+1:end)];
                        rec.info.presentOri = [rec.info.presentOri(1:i-1); rec.info.presentOri(i+1:end)];
                        rec.info.presentImg = [rec.info.presentImg(1:i-1); rec.info.presentImg(i+1:end)];

                        rec.expo.selectTimes = [rec.expo.selectTimes(1:i-1) rec.expo.selectTimes(i+1:end)];
                    end
                end
            end

            %stim size in pixels
            sp = GetEvents(data, data.passes.IDs(data.passes.SlotIDs == 1), 'Op Variable', 'scale stim', 0, 'Destination');
            assert(var(sp) == 0);
            sp = sp(1);
            
            if PARSE_MULTI
                assert(false) %todo stim/rf diameter needs to be subunit based
            end
            
            %hardcoded pixels per degree
            ppd = GetEvents(data, data.passes.IDs(data.passes.SlotIDs == 0), 'Op Variable', 'set pix/deg', 0, 'Destination');
            assert(var(ppd) == 0);

            if str2double(unitName(1:8)) <= 20191118
                %prior to 11/18/19, use 41.88ppd at a 38cm monitor width.
                ppd = 41.88;
            else
                ppd = ppd(1);
            end

            %stimulus and rf size (diameter) in degrees
            rec.expo.stimDiameter = sp/ppd;
            
            %verify stimulus surface values
            checksum_ori = sum(mod(rec.info.presentOri - mod(GetEvents(data, stimPassIDs, 'Surface', 'stim surface', 0, 'Orientation')-90, 360), 360));
            if checksum_ori == 90*length(rec.info.presentOri)
                disp('Surface orientation offset by 90 deg, reassigning stimulus Rotation');
                surfaceRotation = mod(GetEvents(data, stimPassIDs, 'Surface', 'stim surface', 0, 'Orientation')-90, 360);
                rec.info.presentOri = surfaceRotation;
            else
                assert(checksum_ori == 0);
            end

            checksum_texid = sum(rec.info.presentTex - GetEvents(data, stimPassIDs, 'Surface', 'stim surface', 0, 'Texture ID'));
            if checksum_texid > 15
                disp('-texture ID checksum failed');
                surfaceTextureId = GetEvents(data, stimPassIDs, 'Surface', 'stim surface', 0, 'Texture ID');
                if unique(surfaceTextureId) == 1
                    disp('--homogeneous surface textures, reassigning stimulus texture IDs');
                    rec.info.presentTex = surfaceTextureId;
                end
            end
        end
        
        %these things are not handled in the multi contact recordings
        rec.expo.rfDiameter = rec.expo.stimDiameter*.4;

        %check for broken presentations
        pTimes = (rec.expo.presentEndTimes - rec.expo.presentTimes)*1e-4;
        rec.info.presentFail = abs(pTimes - (rec.info.stimDuration)*2) > .002;
        fprintf('-broken trials: %i\n', sum(rec.info.presentFail));

        %% collect and parse stimuli
        fprintf('Building the pyramids...');
        clf;

        %test padding
        N = rec.info.Pyr.resolution;
        S = rec.info.Pyr.scales;
        O = rec.info.Pyr.orientations;
        pad = rec.info.Pyr.padding;


        img = (double(stimData{1}.set(4).img)-N)/N;
        img = padarray(img, [pad pad]);
        img = imresize(img, [N N]);

        [imgSpyr, pind, pyr] = getSpyr3(img, S, O);
        imgSpyr = imgSpyr/max(imgSpyr(:));

        rec.info.Pyr.pind = pind;

        %generate pyramid mask from pind
        m = [];
        for j = 1:length(rec.info.Pyr.pind)
            z = 1;
            if j == 1 && rec.info.Pyr.mask.highres == false
                z = 0;
            end
            if rec.info.Pyr.pind(j,1) == rec.info.Pyr.pind(1,1) && rec.info.Pyr.mask.highscale == false
                z = 0;
            end
            if rec.info.Pyr.pind(j,1) == rec.info.Pyr.pind(end-1,1) && rec.info.Pyr.mask.lowscale == false
                z = 0;
            end
            if j == length(rec.info.Pyr.pind) && rec.info.Pyr.mask.lowres == false
                z = 0;
            end

            mn = z*ones(1,prod(rec.info.Pyr.pind(j,:)));
            m = [m mn];
        end
        %double for on/off
        m = [m m];
        rec.info.Pyr.mask.M = spdiags(m',0, length(m), length(m));
        rec.info.Pyr.mask.M(:,~any(rec.info.Pyr.mask.M,2)) = [];

        %prepare stimulus array
        stimTexSet = unique(rec.info.presentTex);
        stimImgSet = unique(rec.info.presentImg);
        stimOriSet = unique(rec.info.presentOri);

        stimSpyr = cell(length(stimTexSet), length(stimImgSet), length(stimOriSet));
        stimImg = cell(length(stimTexSet), length(stimImgSet), length(stimOriSet));
        stimIsBlank = false(length(stimTexSet), length(stimImgSet), length(stimOriSet));
        for i = 1:length(stimTexSet)
            fprintf('.');
            for j = 1:length(stimImgSet);
                img = (double(stimData{stimTexSet(i)}.set(stimImgSet(j)).img)-128)/128;
                img = padarray(img, [pad pad]);

                for k = 1:length(stimOriSet);
                    %check if blank and flag
                    if (isfield(stimData{1}.set(1), 'isBlank'))
                        stimIsBlank(i,j,k) = stimData{stimTexSet(i)}.set(stimImgSet(j)).isBlank;
                    end
                    %rotate and pyramidize image
                    imgRotate = imrotate(img, stimOriSet(k), 'bicubic', 'crop');
                    imgRotate = imresize(imgRotate, [N N]);
                    stimImg{i, j, k} = imgRotate;
                    [imgSpyr, ~] = getSpyr3(imgRotate, S, O);
                    stimSpyr{i,j,k} = imgSpyr;
                end
            end
        end
        disp('!');
        
        %%
    if VERBOSE
        img = (double(stimData{1}.set(3).img)-128)/128;
        img = padarray(img, [pad pad]);
        img = imresize(img, [N N]);

        [imgSpyr, pind, pyr] = getSpyr3(img, S, O);
        imgSpyr = 1*imgSpyr/max(imgSpyr(:));
        
        r = sum(prod(pind,2));
        imgSpyrOn = imgSpyr;
        imgSpyrOn(r+1:end) = 0;
        imgSpyrOff = imgSpyr;
        imgSpyrOff(1:r) = 0;   
        
        figure;
        colormap(gray);
        imagesc(img);
        title('Image');
        axis equal tight off;

        figure
        subplot(1,2,1);
        spyrDisp3(imgSpyrOn, pind, pyr, true);
        title('sPyr On representation')
        subplot(1,2,2);
        spyrDisp3(imgSpyrOff, pind, pyr, true);
        title('sPyr Off representation')
        
        figure;
        spyrDisp3(imgSpyr, pind, pyr, true);
        title('sPyr representation')
    end

        %% psth and latency calculations
        %do this for every contact
        if ~PARSE_MULTI
            rec.info.contact = 1;     
        else
            rec.info.contact = find(rec.spike.quality(:,1) == 'g');
            %rec.info.contact = rec.info.contact(rec.info.contact<=718 & rec.info.contact>=450);
            rec.info.contact = rec.info.contact(rec.info.contact<=300);
        end
        %construct response vectors using known latency
        responseSet = nan(length(rec.info.contact),length(rec.expo.spikeWide));
        responseMetric.baseline = nan(length(rec.info.contact),1);
        responseMetric.stimPrefPeak = nan(length(rec.info.contact),1);
        responseMetric.stimAveragePeak = nan(length(rec.info.contact),1);
        
        for c = 1:length(rec.info.contact)
            disp(['Calculating response latency: c=' num2str(rec.info.contact(c)) '...']);
            spikeSet = rec.expo.spike(:,rec.info.contact(c));
            spikeWideSet = rec.expo.spikeWide(:,rec.info.contact(c));
            %spikeChunkSet = rec.expo.spikeChunk;

            rate = nan(size(spikeSet));
            for i = 1:length(rate)
                spk = spikeSet{i};        
                rate(i) = numel(spk);
            end
            rateSelect = median(rate)+1;

            gx = linspace(-1,1,2e4+1);
            fg = @(x) exp(-1*gx.^2/(2*((x/10)^2)));
            spkFilter = @(x)(fg(x)/sum(fg(x)));

            %psth for every presentation
            psthWide = zeros(length(rec.expo.spikeWideTime),1);
            for i = 1:length(spikeWideSet)
                    idex = floor(spikeWideSet{i}*1e4)+1;
                    idex = idex(idex <= length(rec.expo.spikeWideTime));
                    psthWide(idex) = psthWide(idex) + 1;
            end
            psthWide = 1e4*psthWide / length(spikeWideSet);
            psthWideFilt = conv(psthWide, spkFilter(.05), 'same');

            %psth for responses over median
            psthWideSelect = zeros(length(rec.expo.spikeWideTime),1);
            for i = 1:length(spikeWideSet)
                if rate(i) >= rateSelect;
                    idex = floor(spikeWideSet{i}*1e4)+1;
                    idex = idex(idex <= length(rec.expo.spikeWideTime));
                    psthWideSelect(idex) = psthWideSelect(idex) + 1;
                end
            end
            psthWideSelect = 1e4*psthWideSelect / sum(rate >= rateSelect);
            psthWideSelectFilt = conv(psthWideSelect, spkFilter(.015), 'same');

            %calculate baseline rate
            if ~isfield(stimData{1}.set(1), 'isBlank')
                
                %psth for start of presentation block
                psthChunk = zeros(length(rec.expo.spikeChunkTime),1);
                for i = 1:length(rec.expo.spikeChunk)
                        idex = floor(rec.expo.spikeChunk{i}*1e4)+1;
                        idex = idex(idex <= length(rec.expo.spikeChunkTime));
                        psthChunk(idex) = psthChunk(idex) + 1;
                end
                psthChunk = 1e4*psthChunk / length(rec.expo.spikeChunk);
                psthChunkFilt = conv(psthChunk, spkFilter(.03), 'same');

                %analyze psth of chunk before stim onset
                baselinePsth = psthChunkFilt(rec.expo.spikeChunkTime > -.400 & rec.expo.spikeChunkTime < 0);
                rec.info.baselineRate(c) = mean(baselinePsth);

                psthBaseline = psthChunkFilt;
                psthBaselineTime = rec.expo.spikeChunkTime;

                fprintf('-no blanks, analyzing pre-stimulus period');
            else
                psthBlank = zeros(length(rec.expo.spikeWideTime),1);
                blankCounter = 0;
                for i = 1:length(spikeWideSet)
                    isblank = stimIsBlank(stimTexSet == rec.info.presentTex(i), stimImgSet == rec.info.presentImg(i), stimOriSet == rec.info.presentOri(i));
                    if isblank
                        blankCounter = blankCounter + 1;
                        idex = floor(spikeWideSet{i}*1e4)+1;
                        idex = idex(idex <= length(rec.expo.spikeWideTime));
                        psthBlank(idex) = psthBlank(idex) + 1;
                    end
                end

                if blankCounter == 0
                    %analyze psth of chunk before stim onset
                    baselinePsth = psthChunkFilt(rec.expo.spikeChunkTime > -.400 & rec.expo.spikeChunkTime < 0);
                    rec.info.baselineRate(c) = mean(baselinePsth);

                    psthBaseline = psthChunkFilt;
                    psthBaselineTime = rec.expo.spikeChunkTime;

                    fprintf('-no valid trials were blanks, analyzing pre-stimulus period');
                else
                    psthBlank = 1e4*psthBlank / blankCounter;
                    psthBlank = conv(psthBlank, spkFilter(.03), 'same');

                    rec.info.baselineRate(c) = mean(psthBlank(rec.expo.spikeWideTime > 0.000 & rec.expo.spikeWideTime < 0.400));

                    psthBaseline = psthBlank;
                    psthBaselineTime = rec.expo.spikeWideTime;
                    fprintf(['-analyzing ' num2str(blankCounter) ' blank trials']);
                end
            end
            fprintf(', baseline rate: %2.2f spk/sec\n', rec.info.baselineRate(c));

            %look at wide times and see when it rises above threshold for latency
            signal = psthWideSelectFilt - rec.info.baselineRate(c);
            signal(rec.expo.spikeWideTime <= 0 | rec.expo.spikeWideTime > 0.400) = 0;
            signalRise = signal > 0.3*(max(signal));
            responseLatency = rec.expo.spikeWideTime(find(signalRise, 1, 'first'));

            if isempty(responseLatency)
                disp('-failed to detect response latency, assuming default.')
                responseLatency = .060;
            elseif responseLatency < 0.020 || responseLatency > 0.150
                disp('-failed to detect response latency, assuming default.')
                responseLatency = .060;
            end

            fprintf('-lantency: %2.2f ms.\n', responseLatency*1e3)
            rec.info.responseLatency(c) = responseLatency;

            if PARSE_MULTI 
                pdfname = [animalName unitName '_' sprintf('%03i', rec.info.contact(c))];
            else
                pdfname = expData{e,6};
            end
            
            if isfield(rec, 'spike')
                h = figure('position', [100 100 900 1000]);
                subplot(2,1,1);
            else
                h = figure('position', [100 100 900 600]);
            end
            %mark latency as offset and pass on to next section
            lh(3) = phyplot(rec.expo.spikeWideTime*1e3, psthWideSelectFilt, 'g-', 'lineWidth', 1, 'color', [0 .75 0]); hold on;
            lh(1) = plot(psthBaselineTime*1e3, psthBaseline, 'LineWidth', 1, 'Color', 0.5*[1 1 1]); 
            lh(2) = plot(rec.expo.spikeWideTime*1e3, psthWideFilt, 'k-', 'LineWidth', 1);
            plot([1 1] * responseLatency*1e3, [0 max(psthWideSelectFilt)*1.1], 'r-');
            plot([responseLatency responseLatency + .200]*1e3, [1 1]*max(psthWideSelectFilt)*1.1, 'r-');
            plot([1 1] * 0*1e3, [0 max(psthWideSelectFilt)*1.05], 'k-');
            plot([0 0 + .200]*1e3, [1 1]*max(psthWideSelectFilt)*1.05, 'k-');

            xlabel('Stimulus onset (ms)');
            ylabel('Response (spk/s)');
            legend(lh, {'Blank trials', 'Average stimuli', 'Preferred stimuli'});
            title(['Cell: ' pdfname], 'interpreter', 'none');
            
            if isfield(rec, 'spike')
                subplot(2,1,2);
                phyplot(rec.spike.waveforms(rec.info.contact(c),:))
                title(sprintf('Depth from tip: %2.2fum', rec.spike.depth(rec.info.contact(c))));
            end
            
            drawnow();
            set(h,'Units','Inches');
            pos = get(h,'Position');
            set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            saveas(h, [figDir 'psth_' pdfname '.pdf'], 'pdf');
            close(h);
            
            %compute measures of psth
            baselineRate = mean(psthBaseline(psthBaselineTime > rec.info.responseLatency(c) & psthBaselineTime < rec.info.responseLatency(c) + rec.info.stimDuration));            
            stimPeakRate = max(psthWideFilt(rec.expo.spikeWideTime > rec.info.responseLatency(c) & rec.expo.spikeWideTime < rec.info.responseLatency(c) + rec.info.stimDuration));            
            stimPrefPeakRate = max(psthWideSelectFilt(rec.expo.spikeWideTime > rec.info.responseLatency(c) & rec.expo.spikeWideTime < rec.info.responseLatency(c) + rec.info.stimDuration));            
            
            %gather response metrics
            responseMetric.baseline = baselineRate;
            responseMetric.stimPrefPeak = stimPrefPeakRate;
            responseMetric.stimAveragePeak = stimPeakRate;
            
            for i = 1:size(responseSet,2)
                %be sure to align spike times to spike window
                spikeTimes = rec.expo.spikeWide{i,rec.info.contact(c)} + rec.expo.spikeWideTime(1);

                %take the spike rate up to stimDuration after latency calculation 
                responseSet(c,i) = sum(spikeTimes >rec.info.responseLatency(c)  & spikeTimes < rec.info.responseLatency(c) + rec.info.stimDuration)/rec.info.stimDuration - rec.info.baselineRate(c);
  
            end
            
            %store metrics
            rec.info.responseMetric(c) = responseMetric;
        end

        %% perform a low-dimensional sampling of spyr representations
        fprintf('Building downsampling matrices');

        % construct a hexagonal tiling
        N = rec.info.Pyr.resolution;
        hScale = rec.info.Ds.scale;
        hRadius = rec.info.Ds.radius;
        Rad3Over2 = sqrt(3) / 2;

        [Ci, Ri] = meshgrid(0:1:N-1);
        n = size(Ri,1);
        C = Rad3Over2 * Ci;
        R = Ri + repmat([0 0.5],[n,n/2]);

        R = R - (N-1)/2;
        C = C - Rad3Over2 - (N/2);

        R = R * hScale;
        C = C * hScale;

        radius = sqrt(R.^2 + C.^2);

        MdsUnmasked = sparse(0,0);

        % for every band
        for z = 1:length(rec.info.Pyr.pind);
            BandRes = rec.info.Pyr.pind(z,1);
            [Cn, Rn] = meshgrid(0:1:BandRes-1);
            hRadiusScale = hRadius * (BandRes/N);
            
            if BandRes <= 16
                %for small bands, just pass through, don't downsample
                MdsBand = zeros(rec.info.Pyr.pind(z,1)^2, length(Rn));
                for i = 1:length(Rn(:))
                    I = Rn == Rn(i) & Cn == Cn(i);
                    radius = sqrt((Rn(i)-max(Rn(end)/2)).^2 + (Cn(i)-max(Cn(end)/2)).^2);
                    if radius < hRadiusScale || BandRes <= 16
                        MdsBand(:,i) = I(:);
                    end
                end
                
            else
                %for larger bands, downsample on hexagonal grid
                sf = 1;
                R1 = sf*R;
                C1 = sf*C;  
                radius = sqrt(R1.^2 + C1.^2);

                Rf = R1(radius <= hRadiusScale)+BandRes/2;
                Cf = C1(radius <= hRadiusScale)+BandRes/2;

        %         plot(Cf, Rf, 'g*');
        %         axis([0 1 0 1]*128 );
        %         pause;

                %for every c/r pair, construct low-res approx
                MdsBand = nan(rec.info.Pyr.pind(z,1)^2, length(Rf));
                sigma = sf*(hScale)*(Rad3Over2/sqrt(2))*.8;
                for i = 1:length(Rf)
                    I = exp(-(Rn-Rf(i)).^2/(sigma^2)).*exp(-(Cn-Cf(i)).^2/(sigma^2));
                    I(I<1e-4) = 0;
                    MdsBand(:,i) = I(:);
                end

                if z == 1
                    ImgC = Cf;
                    ImgR = Rf;
                    MdsI = MdsBand;
                end
            end
            
            %add band onto full matrix;
            MdsUnmasked(end+1:end+size(MdsBand,1),end+1:end+size(MdsBand,2)) = MdsBand;
            fprintf('.');
        end
        fprintf('!\n');

        %double Mds for on/off bands
        MdsUnmasked = kron(eye(2), MdsUnmasked);

        %apply mask to downsampling matrix
        Mds = rec.info.Pyr.mask.M * rec.info.Pyr.mask.M' * MdsUnmasked;
        Mds(:,full(sum(Mds,1))==0) = [];

        %normalize each downsaple
        MdsNorm = bsxfun(@times, Mds, 1./sum(Mds,1));
        
%%
        figure('position', [15          87        1913        1151]);
        spyr = stimSpyr{1,1,1};
        spyrDs = (spyr' * Mds * Mds')';
        subplot(2,3,1);
        spyrDisp3(spyr/max(abs(spyr(:))), pind, pyr, false);
        subplot(2,3,4);
        spyrDisp3(spyrDs/max(abs(spyrDs(:))), pind, pyr, false);
        
        rvals1 = (rand(1,size(Mds,1))-0.5).^3;
        rvals2 = (rand(1,size(Mds,2))-0.5).^3;
        
        subplot(2,3,2);
        spyrDisp3(rvals1, pind, pyr, false);
        
        rDsMap = rvals2 * Mds';
        subplot(2,3,5);
        spyrDisp3(rDsMap, pind, pyr, false);
        
        
        rempty = zeros(1,size(Mds,2));
        rempty(size(Mds,2)/2 +1) = .1;
        rempty(3) = .1;
        rempty(3+18) = 0.2;
        
        rempty(size(Mds,2)/2 +1000-2) = .4;
        rempty(1000) = .4;
        
        subplot(2,3,3);
        spyrDisp3(rempty* Mds', pind, pyr, false);
        
        

    %%

        Ax = sparse(0,0);
        Ay = sparse(0,0);

        % for every band
        for z = 1:length(rec.info.Pyr.pind);
            BandRes = rec.info.Pyr.pind(z,1);

            % Build matrix B
            r2 = 2*ones(BandRes,1);
            r = -ones(BandRes-1,1);
            B = diag(r2,0) + diag(r,1) + diag(r,-1);

            % Sparse matrix B
            B = sparse(B);

            % Build sparse identity matrix
            I = speye(BandRes);

            % Build tridiagonal block matrix A
            ax = kron(B,I);
            ay = kron(I,B);

            %append band derivative to whole matrix
            Ax(end+1:end+size(ax,1),end+1:end+size(ax,2)) = ax;
            Ay(end+1:end+size(ay,1),end+1:end+size(ay,2)) = ay;
        end

        %double deriv matrices for on/off bands
        Ax = kron(eye(2), Ax);
        Ay = kron(eye(2), Ay);

        % u = toyNeuron.cell{1}.sta;
        % spyrDisp(abs(Ay*u)+abs(Ay*u));

        %view for testing
        if VERBOSE
            close all;
            figure('position', [54         193        1498         937]);

            rs = randi(100);
            Tspyr = stimSpyr{rs};
            Timg = stimImg{rs};

            colormap(gray(256));
            subplot(3,2,1);
            imagesc(0:N-1,0:N-1,128+128*Timg/max(Timg(:)));
            hold on;
            %plot(Cf,Rf, 'g.', 'markerSize', 1);
            axis equal tight;

            subplot(3,2,2);
            TimgSmoothed = reshape((Timg(:)'*MdsI)*MdsI', [N N]);
            imagesc(TimgSmoothed);
            hold on;
            %plot(C,R, 'ko');
            axis equal off;

            subplot(3,2,3);
            spyrDisp3(Tspyr/max(abs(Tspyr(:))), pind, pyr, 0);
            title('spyr');

            subplot(3,2,4);
            TspyrDs = (Tspyr'*Mds*Mds')';
            TspyrDs = TspyrDs / max(TspyrDs(:));
            spyrDisp3(TspyrDs, pind, pyr, 0);
            title('DS spyr');

            subplot(3,2,5);
            r = sum(prod(pind,2));
            %below is : v' * XDirSmootherMatrix * upsample
            xDir = (abs(Tspyr'*Mds * (Mds' * Ax * Mds)) * Mds')';
            xDir(r+1:end) = -1*xDir(r+1:end);
            spyrDisp3(xDir/max(xDir(:)), pind, pyr, 0);
            title('d/dx')
            subplot(3,2,6);
            yDir = (abs(Tspyr'*Mds * (Mds' * Ay * Mds)) * Mds')';
            yDir(r+1:end) = -yDir(r+1:end);
            spyrDisp3(yDir/max(yDir(:)), pind, pyr, 0);
            title('d/dy');
        end

        %store DS matrices
        rec.info.Ds.Mds = Mds;
        rec.info.Ds.MdsI = MdsI;
        %rec.info.Ds.Ax = Ax;
        %rec.info.Ds.Ay = Ay;

        %% gather stimuli
        %responselist is buggered
        %step through all stimuli and collect unique stimuli with mean response
        z = 0;
        coordiList = nan(1,size(responseSet,2));
        coordjList = nan(1,size(responseSet,2));
        coordkList = nan(1,size(responseSet,2));
        responseList = nan(length(rec.info.contact),size(responseSet,2));
        responseStdList = nan(length(rec.info.contact),size(responseSet,2));
        responseRepList = nan(length(rec.info.contact),size(responseSet,2));
        for i = 1:length(stimTexSet)
            for j = 1:length(stimImgSet);
                for k = 1:length(stimOriSet);
                    stimMask = rec.info.presentImg == stimImgSet(j) & ...
                               rec.info.presentOri == stimOriSet(k) & ...
                               rec.info.presentTex == stimTexSet(i) & ...
                               ~rec.info.presentFail';

                    if sum(stimMask) > 0  
                        %if blank, disregard
                        if ~stimIsBlank(i,j,k)
                            z = z + 1;

                            coordiList(z) = i;
                            coordjList(z) = j;
                            coordkList(z) = k;
                            responseList(:,z) = mean(responseSet(:,stimMask),2);
                            responseStdList(:,z) = std(responseSet(:,stimMask), [], 2);
                            responseRepList(:,z) = sum(stimMask);
                        end
                    end
                end
            end
        end
        coordiList = coordiList(1:z);
        coordjList = coordjList(1:z);
        coordkList = coordkList(1:z);
        responseList = responseList(:,1:z);
        responseStdList = responseStdList(:,1:z);
        responseRepList = responseRepList(:,1:z);

        %save stimulus data, downsampled
        stimSpyrList = nan(z, size(Mds,2), size(stimSpyr{1},2));
        stimImgList = nan(z, size(MdsI,2));
        for zi = 1:z;
            stimSpyrList(zi,:,:) = Mds' * stimSpyr{coordiList(zi), coordjList(zi), coordkList(zi)};
            stimImgList(zi,:) = MdsI' * stimImg{coordiList(zi), coordjList(zi), coordkList(zi)}(:);
        end

        rec.info.stimSpyrList = stimSpyrList;
        rec.info.stimImgList = stimImgList;
        rec.info.responseList = responseList;
        rec.info.responseStdList = responseStdList;
        rec.info.responseRepList = responseRepList;
        
        fprintf('-presentations: %i, unique: %i\n', length(responseSet), length(responseList));

        
        %% load and append sf/ori data
        disp('Analyzing Sf/Ori data...');

        %if experiment is present, load and process
        fdirname = [dataDir animalName '\' animalName unitName '\'];
        fileName    = [animalName '*[timFixStim].xml'];
        
        files       = dir([fdirname fileName]);
        if length(files)>=1
            fullPath    = [fdirname files(1).name];
            gratingData = GetSfOriDataFromXML(fullPath);
            rec.expo.expOriSfdata = gratingData;

            dimSf = gratingData.spatialFreq;
            dimOri = gratingData.orientation;
            dimPhase = gratingData.phase;
            dimReps = size(gratingData.spikes,4);

            %calculate responses
            rateMean = nan(length(dimSf), length(dimOri), length(dimPhase));
            rateVar = nan(length(dimSf), length(dimOri), length(dimPhase));
            rateReps = nan(length(dimSf), length(dimOri), length(dimPhase));
            for s = 1:length(dimSf);
                for o = 1:length(dimOri);
                    for p = 1:length(dimPhase);

                        sv = gratingData.spikes(s,o,p, squeeze(gratingData.spikeMask(s,o,p,:)));

                        reps = length(sv);
                        rateReps(s,o,p) = reps;
                        rateEst = nan(1, reps);
                        for r = 1:reps;
                            rateEst(r) = sum(sv{r} > rec.info.responseLatency * 1e3 & sv{r} <= (rec.info.responseLatency + rec.info.stimDuration)*1e3) / rec.info.stimDuration;
                        end

                        rateMean(s,o,p) = mean(rateEst);
                        rateVar(s,o,p) = var(rateEst);
                    end
                end
            end

            rec.info.dataOriSf.dimSf = dimSf;
            rec.info.dataOriSf.dimOri = dimOri;
            rec.info.dataOriSf.dimPhase = dimPhase;
            rec.info.dataOriSf.dimReps = dimReps;
            rec.info.dataOriSf.rateMean = rateMean;
            rec.info.dataOriSf.rateVar = rateVar;
            rec.info.dataOriSf.rateReps = rateReps;
        end

        %% load texture data
        addpath('C:\Users\Oleskiw\Google Drive\wrx\Program\V2DropletExperiment\analysis');
        disp('Analyzing Texture data...');

        %if experiment is present, load and process
        fdirname = [dataDir animalName '\' animalName unitName '\'];
        fileName    = [animalName '*[tex5demo].xml'];
        files       = dir([fdirname fileName]);
        if length(files) >= 1

            fullPath    = [fdirname files(1).name];
            textureData = GetTextureDataFromXML(fullPath);
            rec.expo.expTexturedata = textureData;

            dimTexid = textureData.textureFam;
            dimNatNoise = textureData.natNoiseInd;
            dimSeed = textureData.seed;
            dimReps = size(textureData.spikes,4);

            %calculate responses
            rateMean = nan(length(dimTexid), length(dimNatNoise), length(dimSeed));
            rateVar = nan(length(dimTexid), length(dimNatNoise), length(dimSeed));
            rateReps = nan(length(dimTexid), length(dimNatNoise), length(dimSeed));
            for t = 1:length(dimTexid);
                for n = 1:length(dimNatNoise);
                    for s = 1:length(dimSeed);

                        sv = textureData.spikes(t,n,s, squeeze(textureData.spikeMask(t,n,s,:)));

                        reps = length(sv);
                        rateReps(t,n,s) = reps;
                        rateEst = nan(1, reps);
                        for r = 1:reps;
                            rateEst(r) = sum(sv{r} > rec.info.responseLatency * 1e3 & sv{r} <= (rec.info.responseLatency + rec.info.stimDuration)*1e3) / rec.info.stimDuration;
                        end

                        rateMean(t,n,s) = mean(rateEst);
                        rateVar(t,n,s) = var(rateEst);
                    end
                end
            end

            rec.info.dataTexture.dimTexid = dimTexid;
            rec.info.dataTexture.dimNatNoise = dimNatNoise;
            rec.info.dataTexture.dimSeed = dimSeed;
            rec.info.dataTexture.dimReps = dimReps;
            rec.info.dataTexture.rateMean = rateMean;
            rec.info.dataTexture.rateVar = rateVar;
            rec.info.dataTexture.rateReps = rateReps;
        end

        %castrate unwanted fields
        rec.expo.spike = [];
        rec.expo.spikeWide = [];
        
        
        %saving
        disp(['Saving record: ' animalName unitName setName]);
        save([parsedDir animalName unitName setName], 'rec', '-v7.3');
    else
        %disp(['Loading parsed file: ' animalName unitName setName]);
        %load([parsedDir animalName unitName setName]);
        disp(['Skipping parsed file: ' animalName unitName setName]);
    end
end

%%
disp('Making PSTH PDF...');
psthFilename = 'archive_psth.pdf';
cmdToRun = ['"C:\Program Files (x86)\PDFtk Server\bin\pdftk.exe" psth*.pdf output ' psthFilename];
system(['cd "' figDir '"' '& ' cmdToRun]);
disp(['Wrote to ' psthFilename]);

