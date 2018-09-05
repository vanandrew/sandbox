function val = Laguerre2D(dim,j,a,r);
%
%  img = Laguerre2D(dim,j,a,[r]);
%
%  Generate a Laguerre-Gauss function of order j, with width parameter
%  a.  Pass the optional r argument if you want to use your own 
%  coordinate system.
%
%  Laguerre-Gauss functions are used to approximate the  Hotelling observer.
%  See reference "" for information.
%
%  dim = size of resultant image
%  j   = order of LG functions
%  a   = width parameter of LG functions
%  r   = (optional) matrix defining coordinate system.  This must
%        match the size specified by dim
%
%  Examples: img = Laguerre2D([128,128],2,30);
%            mesh(img);
%            [X,Y] = meshgrid(linspace(-10,10,128),linspace(-10,10,128));
%            r = sqrt(X.^2+Y.^2);
%            img = Laguerre2D([128,128],2,3,r);
%            mesh(linspace(-10,10,128),linspace(-10,10,128),img);
%

if (length(dim) == 1)
  dim = [dim dim];
end

if (nargin == 3)
  [X,Y] = meshgrid((1:dim(2))-(dim(2)+1)/2,(1:dim(1))-(dim(1)+1)/2);
  r = sqrt(X.^2 + Y.^2);
end

  
val = (1/sqrt(2*pi))*2*sqrt(pi)/a * exp(-(pi*r.^2)/(a^2)).*Laguerre1D(j,2*pi*r.^2/(a^2));
