function [tpf,fpf]=DataRoc(tout,fout,dir);
%
%  [tpf,fpf]=DataRoc(tout,fout,dir);
%
%  Uses the data as the cutpoints
%
%  Computes the ROC operating points (by simple thresholding) using
%  the tout and fout decision variable data.
%
%  Returns:
%      tpf   = True-positive fraction values
%      fpf   = False-positive fraction values
%  Inputs:  
%      tout  = true decision variable data
%      fout  = false decision variable data
%      dir   =  1 (default) implies that large values are abnormal
%              -1 implies that small values are abnormal
%
%    NOTE: Az can be computed by trapz(fpf,tpf) although it is sometimes
%          necessary to flip the order of the tpf and fpf values, i.e.,
%          trapz(fliplr(fpf),fliplr(tpf)).
%

if nargin==2,
  dir=1;
end;

data=[tout(:) ; fout(:)];
minn=min(data);
maxx=max(data);
xs = sort(data);
xs = unique(xs);

%xs=linspace(minn,maxx,500);

if dir ==1,
  for i=1:length(xs),
    tpf(i)=sum(tout > xs(i))/length(tout);
    fpf(i)=sum(fout > xs(i))/length(fout);
  end;
else
  for i=1:length(xs),
    tpf(i)=sum(tout < xs(i))/length(tout);
    fpf(i)=sum(fout < xs(i))/length(fout);
  end;
end

