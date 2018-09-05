function img = CircSignal(dim,radius,mag,pos)
%
% img = CircSignal(dim,radius,mag,[pos])
%
% Generate a circular disk signal image with dimension 'dim', 
% radius in pixels, and mag specifying the gray level value.
%
% 'pos' is an optional argument specigying the position of the
% center of the circ.  The center of the image is default.
%
% See Also: GaussianSignal
%

if (length(dim) == 1)
  dim = [dim dim];
end

if (nargin == 3)
  % the flip is because we use (x,y) coordinates
  pos = fliplr((dim - [1 1]) / 2);
end

x = 0:(dim(2)-1);
y = 0:(dim(1)-1);
[X,Y]=meshgrid(x,x);

r = sqrt((X-pos(1)).^2 + (Y-pos(2)).^2);
img = mag * (r<=radius);