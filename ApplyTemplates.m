function v = ApplyTemplates(u,g)
%
% v = ApplyTemplates(u,g)
%
% Apply the templates u to the images g.
%
% Inputs:
%    u  - [NPixels X NTemplates] Contains NTemplates templates that are
%         to be applied to the images g.
%    g  - [NPixels X NImages] The images
%
% Outputs:
%    v  - [NTemplates X NImages] the template outputs for each image.
%

[NP,NT]  = size(u);
[NP1,NI] = size(g);

if (NP1 ~= NP) 
  disp('Numbers of pixels do not match.');
  return
end

% I don't know why I even wrote a function for this thing!
v = u'*g;