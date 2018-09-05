function val = Laguerre1D(j,x)
%
% val = Laguerre1D(j,x)
%
% Compute the 1-D Laguerre-Gauss function of order j at values defined
% by x
%

val = zeros(size(x));
for jp = 0:j,
  val = val + ((-1)^jp) * (prod(1:j)/((prod(1:jp)*prod(1:(j-jp)))))*...
      (x.^jp)/(prod(1:jp));
end
