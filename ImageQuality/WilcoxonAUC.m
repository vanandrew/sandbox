function [AUC,tpf,fpf]=WilcoxonAUC(Sout,Nout);
%
%  [AUC,tpf,fpf]=WilcoxonAUC(Sout,Nout);
%
%  Computes the AUC, and the ROC operating points (by simple 
%  thresholding) using the Sout and Nout decision variable data.
%
%  Returns:
%      AUC   = Wilcoxon area under the ROC curve
%      tpf   = True-positive fraction values
%      fpf   = False-positive fraction values
%  Inputs:  
%      Sout  = with-signal decision variable data
%      Nout  = without-signal decision variable data
%
%  NOTE: --IMPORTANT-- this routine assumes that the signal decision
%  variable data is (generally) larger than the noise, ie, large decision
%  variable outputs correspond to high confidence of signal present.
%

data=[Sout(:) ; Nout(:)];
xs = sort(data);
xs = unique(xs);

tpf = zeros(length(xs)+1,1);
fpf = zeros(length(xs)+1,1);

cnt = 1;
% go backwards so that the ROC curve starts at (0,0).
for thresh = fliplr(xs'),
  tpf(cnt)=sum(Sout > thresh)/length(Sout);
  fpf(cnt)=sum(Nout > thresh)/length(Nout);
  cnt = cnt+1;
end  
tpf(cnt) = 1;
fpf(cnt) = 1;

AUC = trapz(fpf,tpf);
