echo on;
% This is a quick runthrough to familiarize you with some of the
% code in this toolbox.
%
% Let's start by generating some images.
%
% The MVNLumpy routine will sample a multivariate (correlated) Gaussian
% to generate the specified number of images (100 in this case).  This 
% will take a little time.
%
n = MVNLumpy(10,zeros(128,128),ones(128,128),100);

% And let's display them
echo off;
for i = 1:4,
  subplot(2,2,i),imagesc(reshape(n(:,i),128,128));colormap(gray);
end
echo on;

% press <return> to continue
pause;

% The signal images will have the same background but will have a Gaussian
% blob added to each one.

s = MVNLumpy(10,zeros(128,128),ones(128,128),100);

sig = GaussianSignal(128,5,.2);
sig =sig(:);

s = s + sig(:,ones(100,1));

% and display
echo off;
for i = 1:4,
  subplot(2,2,i),imagesc(reshape(n(:,i),128,128));colormap(gray);
end
echo on;

% press <return> to continue
pause;

% We are going to use the Laguerre-Gauss templates for our observer.

U0 = Laguerre2D(128,0,15);
U1 = Laguerre2D(128,1,15);
U2 = Laguerre2D(128,2,15);
U3 = Laguerre2D(128,3,15);
U4 = Laguerre2D(128,4,15);

% and display the first four
echo off;
subplot(2,2,1),imagesc(U0);colormap(gray);
subplot(2,2,2),imagesc(U1);colormap(gray);
subplot(2,2,3),imagesc(U2);colormap(gray);
subplot(2,2,4),imagesc(U3);colormap(gray);
echo on;

% press <return> to continue
pause;

% now generate the template matrix
U = [U0(:) U1(:) U2(:) U3(:) U4(:)];

% finally, compute the channelized Hotelling outputs
[tS,tN]=CHotelling(s,n,U);

% The tS, and tN are decision variable outputs which we can
% perform ROC analysis on.

[AUC,tpf,fpf]=WilcoxonAUC(tS,tN);

% display the area under the ROC curve
AUC

% plot the ROC curve

clf; plot(fpf,tpf);
echo off;
