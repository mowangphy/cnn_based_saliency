function [grad_mean_norm] = do_saliency(net, im, label, labelVec, ampPara, levelamp, isPost, curSuperPixel, curNSP, lowlevelfeature, isRefine, isSmooth, varargin)

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 35 ;%300
opts.learningRate = 0.001 ;
opts.disableDropout = 1;

opts.contrastNormalization = false ;
opts.continue = false ;
opts.gpus = 1 ;

opts.conserveMemory = true ;
opts.sync = true ;
opts.prefetch = false ;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

opts.backPropDepth = +inf ;

inputLearningRate = 20.0;
decrate = 0.999;
normDGThres = 0.01;
alpha = 200;

% -------------------------------------------------------------------------
%                                                 Network initialization
% -------------------------------------------------------------------------

numGpus = numel(opts.gpus) ;
if numGpus == 1
  net = vl_simplenn_move(net, 'gpu') ;
end

% -------------------------------------------------------------------------
%                                                   Train and validate
% -------------------------------------------------------------------------

rng(0) ;

[nDim1_im, nDim2_im, nDim3, batchSize] = size(im);% 224-224-3-batchSize
im_ori = im;

if numGpus == 1
  one = gpuArray(single(1)) ;
  im = gpuArray(im) ;
else
  one = single(1) ;
end

lr = 0 ;
res = [] ;
obj_value = zeros(1, opts.numEpochs);
bp_time_all = 0;

for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
	
  bp_time = tic ;
  % backprop
  net.layers{end}.class = label ; % 1*batchSize
  net.layers{end}.labelVec = labelVec; % nClasses*batchSize
  net.layers{end}.alpha = alpha;
  res = vl_simplenn_newsaliency(net, im, one, res, ...
                      'accumulate', 0, ...
                      'disableDropout', opts.disableDropout, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync) ;

  % gradient step
  dataGradient = res(1).dzdx; % 224-224-3-batchSize
  dataGradient = dataGradient .* (dataGradient > 0);
  %need to be normalized
  %normalize 
  [nDim1, nDim2, nDim3, nDim4] = size(dataGradient);%for mnist, nDim3 = 1;
   
  tmpMax = max(dataGradient, [], 1);
  maxElem = max(tmpMax, [], 2);
  tmpMin = min(dataGradient, [], 1);
  minElem = min(tmpMin, [], 2);
  n1 = bsxfun(@minus, dataGradient, minElem);
  n2 = bsxfun(@minus, maxElem, minElem);
  normDG = bsxfun(@rdivide, n1, n2);
  % normDG = dataGradient;
	
  im = im - inputLearningRate * normDG;
  
  %fprintf('Saving after every epoch');
  if (epoch == opts.numEpochs)
	revised_im = im;
	revised_im = gather(revised_im);
	
	% im_ori = im_ori./255;
	% revised_im = revised_im./255;
	gradient_hengyue = im_ori - revised_im;
	
	tmpMax = max(gradient_hengyue, [], 1);
	maxElem = max(tmpMax, [], 2);
	tmpMin = min(gradient_hengyue, [], 1);
	minElem = min(tmpMin, [], 2);
	n1 = bsxfun(@minus, gradient_hengyue, minElem);
	n2 = bsxfun(@minus, maxElem, minElem);
	gradient_hengyue_norm = bsxfun(@rdivide, n1, n2);
	
	if (isPost == 1)
		grad_mean_norm = mean(gradient_hengyue_norm, 3);% 224-224-1-batchSize
		grad_mean_norm = squeeze(grad_mean_norm);
		maxGrad = max( max( grad_mean_norm, [], 1 ), [], 2 );
		
		for batch = 1:batchSize
			curMap = grad_mean_norm(:, :, batch);
			level1 = graythresh( curMap );
			level1 = level1 * levelamp;
			level1 = level1 * maxGrad(batch);
			
			curMap = curMap .* (curMap > level1);
			
			if (isSmooth)
				for iSP = 0:(curNSP(batch)-1)
					curSPIdx = find(curSuperPixel(:, :, batch)==iSP);
					curPixels = curMap(curSPIdx);
					curMeanPixels = mean(curPixels);
					curMap(curSPIdx) = curMeanPixels;
				end
			end
			
			grad_mean_norm(:, :, batch) = curMap;
		end
		
		tmpMax = max(grad_mean_norm, [], 1);
		maxElem = max(tmpMax, [], 2);
		tmpMin = min(grad_mean_norm, [], 1);
		minElem = min(tmpMin, [], 2);
		n1 = bsxfun(@minus, grad_mean_norm, minElem);
		n2 = bsxfun(@minus, maxElem, minElem);
		grad_mean_norm = bsxfun(@rdivide, n1, n2);
		
		% lowlevelfeature = lowlevelfeature .* (lowlevelfeature > 0.1);		
		if (isRefine)
			grad_mean_norm = grad_mean_norm .* (lowlevelfeature + 0.2);
		end
		
		tmpMax = max(grad_mean_norm, [], 1);
		maxElem = max(tmpMax, [], 2);
		tmpMin = min(grad_mean_norm, [], 1);
		minElem = min(tmpMin, [], 2);
		n1 = bsxfun(@minus, grad_mean_norm, minElem);
		n2 = bsxfun(@minus, maxElem, minElem);
		grad_mean_norm = bsxfun(@rdivide, n1, n2);
		% one more filter
		% grad_mean_norm = grad_mean_norm .* (grad_mean_norm > 0.2);
		%%%%%%%%%%%%%
		grad_mean_norm = grad_mean_norm .* ampPara;
		idxNorm = find(grad_mean_norm > 1);
		grad_mean_norm(idxNorm) = 1;
		grad_mean_norm = grad_mean_norm .* (grad_mean_norm > 0.2);
	end
	
	%strcmd3_2 = sprintf('imwrite(grad_mean_norm, ''./%d_top_hengyue.png'');', n);
	%eval(strcmd3_2);
  end 
  bp_time = toc(bp_time) ;
  bp_time_all = bp_time_all + bp_time;
  
  inputLearningRate = inputLearningRate * decrate;
end

fprintf(' BP speed %.6f s/img ', bp_time_all/batchSize) ;
fprintf('\n') ;