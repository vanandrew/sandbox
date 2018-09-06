function [dp] = dprime(Sout,Nout)
%
% dp = dprime(Sout,Nous)
%
% Compute the d' (SNR) for the decision variable data given by Sout,
% and Nout -- the signal and noise outputs, respectively
%

dp = sqrt( ((mean(Sout) - mean(Nout)).^2) / (.5*var(Sout) + .5*var(Nout)));
