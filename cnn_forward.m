function [label1, label5, predictions_original] = cnn_forward(net, im, varargin)

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

res = [] ;

batch_time = tic ;
numGpus = numel(opts.gpus) ;
if numGpus == 1
  net = vl_simplenn_move(net, 'gpu') ;
  one = gpuArray(single(1)) ;
  im = gpuArray(im) ;
else
  one = single(1) ;
end
	
res = vl_simplenn(net, im, [], res, ...
                  'accumulate', 0, ...
                  'disableDropout', 1, ...
                  'conserveMemory', opts.conserveMemory, ...
                  'backPropDepth', opts.backPropDepth, ...
                  'sync', opts.sync) ;

% print information
batch_time = toc(batch_time) ;

predictions_original = gather(res(end-1).x) ;%only one image, should be 1*1*nClasses; minibatch: 1*1*nClasses*batchSize
predictions = predictions_original;
batchSize = size(predictions, 4);
predictions = squeeze(predictions); % nClasses*batchSize

label5 = zeros(batchSize, 5);

[Val Idx] = max(predictions,[], 1);
label1 = Idx;
% size(predictions)
% size(Idx)
label5(:, 1) = Idx;
for i = 2:5
	for j = 1:batchSize
        predictions( Idx(j), j ) = -1;
    end
	[Val Idx] = max(predictions,[], 1);
	label5(:, i) = Idx;
end

fprintf('Forward speed: %.6f s/img; ', batch_time/batchSize) ;
% fprintf('\n') ;



