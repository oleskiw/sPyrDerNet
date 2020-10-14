close all;
clear all;
clc;

fName = 'bundle_64_1024_1-2-3_1';
K = 2;
IMG = 2;
PHASED = false;


fNameModel = [fName '_model'];
fNameTargets = [fName '_targets'];
load(fNameModel, 'dataModel')
load(fNameTargets, 'dataTargets');
load(['C:\Users\Oleskiw\Google Drive\wrx\Program\sciTest\' fName '_output.mat']);




imgNumber = dataTargets.imgNumber == IMG;
k = dataTargets.k == K;
phased = dataTargets.phased == PHASED;

[spyr, pind] = getSpyr(squeeze(dataTargets.img(1,:,:)));

set = find(imgNumber & k);


figure('position', [9    48   601   345]);
showSpyr(dataModel.components(:,K), pind, 'auto1'); 

REP = 3;
for s = 1:REP:length(set)
    figure('position', [22         444        1800         800])
    
    for i = 1:size(output.intermediateSpyr, 2)
        clf;
        for r = 0:REP-1

            subplot(REP,5,r*5+1);
            image(128 + 64*putSpyr(dataTargets.imgSpyr(set(s)+r,:)', pind));
            colormap(gray(256));
            axis equal off;

            subplot(REP,5,r*5+2);
            stem(dataTargets.expression(set(s)+r,:), 'b.');
            hold on;
            plot(dataTargets.expressionTarget(set(s)+r,:), 'ko');
            plot(squeeze(output.intermediateExpression(set(s)+r, i, 1, :)), 'r+');

            subplot(REP,5,r*5+3);
            image(128 + 64*putSpyr(squeeze(output.intermediateSpyr(set(s)+r, i, :)), pind));
            colormap(gray(256));
            axis equal off;

            subplot(REP,5,r*5+4);  
            plot(output.intermediateObjective(set(s)+r, :), 'b-');
            hold on;
            plot(i, output.intermediateObjective(set(s)+r, i), 'r+');

            subplot(REP,5,r*5+5);
            image(128 + 64*(putSpyr(dataTargets.imgSpyr(set(s)+r,:)', pind) - putSpyr(squeeze(output.intermediateSpyr(set(s)+r, i, :)), pind)));
            colormap(gray(256));
            axis equal off;
            
        end
        if i < size(output.intermediateSpyr, 2)  
            drawnow;
            pause;
        end
    end
    figure;
    showSpyr(dataTargets.imgSpyr(set(s),:)', pind, 'auto1')
    figure;
    showSpyr(output.imgSpyr(set(s),:)', pind, 'auto1')
    
    drawnow;
    pause;
    
end