% ImageQuality  Toolbox 
% Version 0.9b  Mar-25-2001
%       (c) The University of Arizona
%
% Background Generation
%   LumpyBgnd   - Generate a lumpy background -- type 1
%   MVNLumpy    - Generate a lumpy background -- type 2
%   MVNLumpyConv- Same as above but without the wrap around artifacts
%   CLB         - Generate a clustered-lumpy background as described
%                 by Bochud, et. al.
%   NOTE: CLB and LumpyBgnd require that poissrnd from the Stats toolbox
%         is available.
% 
% Signal Generation
%   CircSignal  - Generate a circular disc signal
%   GaussianSignal - Generate a gaussian blob signal
%
% Observers
%   Laguerre2D  - Generate Laguerre-Gauss functions for approximating
%                 the ideal observer.
%   ApplyTemplates - Apply templates to images
%   CHotelling  - Compute the Channelized hotelling observer outputs
%
% Two-alternative-forced choice
%   Test2AFC    - Perform a 2AFC experiment
%
% ROC
%   WilcoxanAUC - Compute the Wilcoxan area under the ROC curve.
%                 This also returns ROC operating points for graphing
%   dprime      - Compute the d' SNR statistic
%   AzFromDP    - Compute the Az from the d' statistic
% 
% Miscellaneous
%   RunExperiment - demonstration of some of these routines -- try this
%
% This toolbox is a work in progress.  Please report any bugs or suggestions
% to Matthew Kupinski ( kupinski@radiology.arizona.edu ).  
%
