function [tS,tN] = CHotelling(IS,IN,Temps)
%
% [tS,tN] = CHotelling(IS,IN,Temps)
%
% Compute the channelized Hotelling observer outputs for
% the signal-images given by IS, the noise images given by
% IN, and the templates given by Temps.
%
% Inputs: 
%       IS, IN -- [NPixels X NImages] The images (in vector format)
%       Temps  -- [NPixels X NTemplates] The templates
%
% Outputs: tS,tN -- [NImages X 1] -- the decision variable outputs
%                   can use WilcoxanAUC or dprime on this output.
%
% See Also: WilcoxanAUC
%           dprime
%           RunExperiment
%

% Compute channel outputs
vS = ApplyTemplates(Temps,IS);
vN = ApplyTemplates(Temps,IN);

% compute mean difference image -- signal
sbar = mean(IS')-mean(IN');

% Intra-class scatter matrix
S = .5*cov(vN') + .5*cov(vS');

% channel template
wCh = inv(S) * Temps' * sbar';

% apply the channel template to produce outputs
tS = ApplyTemplates(wCh,vS);
tN = ApplyTemplates(wCh,vN);