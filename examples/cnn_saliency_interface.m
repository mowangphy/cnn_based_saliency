clear;
clc;
run(fullfile(fileparts(mfilename('fullpath')), '../../matlab/vl_setupnn.m')) ;

%%
load imdb_pascal.mat; % imdb
[H, W, C, nSamples] = size(imdb.images.data);

imClasses = {'Plane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'};
nClasses = length(imClasses); % Pascal voc 2012

load well_trained_cnn.mat; % load the pre-trained net
net_saliency = net; 
net_saliency.layers{end} = struct('type', 'softmaxloss_original_mb');
net_forward = net;
net_forward.layers{end} = struct('type', 'softmax') ;

load superpixel_maps_for_all_images.mat;
load number_of_superpixels_of_all_images.mat;
load low_level_saliency_features.mat;

clear net;

ampPara = 1.70 
levelamp = 1.30   
nClasses = 20;

isSmooth = 1
isRefine = 1

%%
top5 = cell(1, nSamples);

obj_value_all = [];
nCorrect = 0;
nBG = 0;
correct_list = zeros(1, nSamples);

batchSize = 50;
nBatch = floor(nSamples/batchSize);
nResidual = nSamples - (nBatch*batchSize);

imgIdx = 1;
for i = 1:nBatch
	fprintf('Process the %d th batch, overall %d batches: ', i, nBatch);
	im = imdb.images.data( :, :, :, ((i-1)*batchSize+1):(i*batchSize) );
	label = imdb.images.labels( ((i-1)*batchSize+1):(i*batchSize) );
	
	% ------ forward -----
	[label1, label5, labelVec] = cnn_forward(net_forward, im);
	
	isPost = 1;
	
	curSuperPixel = superpixel_P(:, :, ((i-1)*batchSize+1):(i*batchSize));
	curNSP = nSuperpixel_P(((i-1)*batchSize+1):(i*batchSize));
	
	curlowlevelmap = low_level_saliency_features(:, :, ((i-1)*batchSize+1):(i*batchSize));
	
	tmpMax = max(curlowlevelmap, [], 1);
	maxElem = max(tmpMax, [], 2);
	tmpMin = min(curlowlevelmap, [], 1);
	minElem = min(tmpMin, [], 2);
	n1 = bsxfun(@minus, curlowlevelmap, minElem);
	n2 = bsxfun(@minus, maxElem, minElem);
	curlowlevelmap = bsxfun(@rdivide, n1, n2);
	
	% --------- backward --------	
	[saliency_maps] = do_saliency(net_saliency, im, label1, labelVec, ampPara, levelamp, isPost, curSuperPixel, curNSP, curlowlevelmap, isRefine, isSmooth);
	
	for j = 1:batchSize
		strcmd3_3 = sprintf('imwrite(saliency_maps(:, :, j), ''./top1_results/Oxford_minibatch/%d_top1_Oxford.png'');', imgIdx);
		eval(strcmd3_3);
		imgIdx = imgIdx + 1;
	end
end

% -------------- deal with the residule data ------------------
fprintf('Processing the residule data: ');
im = imdb.images.data( :, :, :, end-nResidual+1:end );
label = imdb.images.labels( end-nResidual+1:end );
% ------ forward -----
[label1, label5, labelVec] = cnn_forward(net_forward, im);

isPost = 1;

curSuperPixel = superpixel_P(:, :, end-nResidual+1:end);
curNSP = nSuperpixel_P(end-nResidual+1:end);
curlowlevelmap = low_level_saliency_features(:, :, end-nResidual+1:end);

tmpMax = max(curlowlevelmap, [], 1);
maxElem = max(tmpMax, [], 2);
tmpMin = min(curlowlevelmap, [], 1);
minElem = min(tmpMin, [], 2);
n1 = bsxfun(@minus, curlowlevelmap, minElem);
n2 = bsxfun(@minus, maxElem, minElem);
curlowlevelmap = bsxfun(@rdivide, n1, n2);

% --------- backward --------	
[saliency_maps] = do_saliency(net_saliency, im, label1, OxfordLabel, ampPara, levelamp, isPost, curSuperPixel, curNSP, curlowlevelmap, isRefine, isSmooth);

for j = 1:nResidual
	strcmd3_3 = sprintf('imwrite(saliency_maps(:, :, j), ''./top1_results/Oxford_minibatch/%d_top1_Oxford.png'');', imgIdx);
	eval(strcmd3_3);
	imgIdx = imgIdx + 1;
end

