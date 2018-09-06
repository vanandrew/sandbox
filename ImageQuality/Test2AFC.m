function Az = Test2AFC(dim,sigI,noiseI,s)
%
% Az = Test2AFC(sigI,noiseI,s)
%
% Perform a simple 2AFC experiment with the images in sigI, and noiseI.
% sigI and noiseI are both (Npixels X Nimages) with image dimensions given
% by dim.  The variable 's' contains a picture of the signal one is trying
% to detect.
%
% This routine will bring three images: the center image contains the
% signal, and the use much choose which of the left or right images contains
% the signal by clicking either on the left or right mouse-button.
%
% Note: Ideally, 2AFC experiments are done under very controlled conditions
% such as lighting, distance to monitor, gray-levels, etc.  This routine
% does not control these situations.  In light of this, this routine exists
% mainly to familiarize one with the 2AFC framework, and shouldn't be used
% in actual experiments.
%

if (length(dim) == 1)
  dim = [dim dim];
end

[nimages,npixels] = size(sigI);

Az = 0;

for i=1:nimages,
  signal = reshape(sigI(:,i),dim);
  noise  = reshape(noiseI(:,i),dim);
  
  flip = rand(1,1) > 0.5;

  if (flip)
    subplot(131),imagesc(signal);colormap(gray);
    subplot(132),imagesc(s);colormap(gray);
    subplot(133),imagesc(noise);colormap(gray);
  else
    subplot(131),imagesc(noise);colormap(gray);
    subplot(132),imagesc(s);colormap(gray);
    subplot(133),imagesc(signal);colormap(gray);
  end    
  truesize;
  t = waitforbuttonpress;
  if (t == 1) break;end;
  
  selection = [];
  
  type = get(gcf,'SelectionType');
  if (strcmp(type,'normal'))
    selection = 0;
  elseif (strcmp(type,'alt'))
    selection = 1;
  else
    disp('Invalid selection');
    selection = -1;
  end;
  
  if ( ((selection == 0) & (flip)) | ...
        ((selection == 1) & (~flip)) )
    fprintf('Correct\n');
    Az = Az + 1;
  else
    fprintf('Incorrect\n');
  end
end
  
close;

Az = Az / nimages;