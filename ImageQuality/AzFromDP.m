function Az = AzFromDP(dp)
%
% Az = AzFromDP(dp)
%
% Compute the Az from the dprime value specified by dp.  This assumes a
% binormal model.
%

Az = .5+.5*erf(dp/2);