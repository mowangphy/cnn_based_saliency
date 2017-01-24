function [Y] = vl_nnsoftmaxloss_newsaliency_original_obj(X,c,oriLabel,alpha,dzdy)

sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

% index from 0
c = c - 1 ;

if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
  c = repmat(c, [sz(1) sz(2)]) ;
else
  % one label per spatial location
  sz_ = size(c) ;
  assert(isequal(sz_, [sz(1) sz(2) 1 sz(4)])) ;
end

% convert to indeces
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * c(:)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;
  
%save c_.mat c_;

% compute softmaxloss
Xmax = max(X,[],3) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;%forward
%save X.mat X;

n = sz(1)*sz(2) ;
if nargin <= 2
  t = Xmax + log(sum(ex,3)) - reshape(X(c_), [sz(1:2) 1 sz(4)]) ;
  Y = sum(t(:)) / n ;
  obj = 0;
else
  %-----------------new saliency strategy original 1---------
  %fprintf('New saliency strategy ... ');
  % calculate the objective function
  Y = X;
  target = oriLabel;
  
  % target(c_) = 0;
  Y = alpha * (Y - target);
  Y(c_) = 1;
  Y = Y * (dzdy / n) ;
  %----------------------------------------------------------
  
  %-----------------new saliency strategy original 1---------
  % fprintf('New saliency strategy ... ');
  % Y = X;
  % target = oriLabel;
  % minLabel = min(oriLabel);
  
  % if (minLabel <= 0)
	% target(c_) = minLabel * 10;
  % else
    % target(c_) = minLabel / 10;
  % end
  
  % Y = Y - target;
  % Y = Y * (dzdy / n) ;
  %----------------------------------------------------------
end