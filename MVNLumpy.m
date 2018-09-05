function img = MVNLumpy(lump_width,muimg,sigimg,numimgs)
%
%  img = MVNLumpy(lump_width,muimg,sigimg,[numimgs])
%
%  Samples an mutlivariate normal distribution with mean given by 
%  muimg, standard deviation given by sigimg, and with a correlation 
%  structure defined by the lump_width.  This method assumes that the
%  autocorrelation function is stationary and Gaussian shaped with a 
%  standard deviation of lump_width.  A Fourier transform is used
%  to convolve white noise with the correlation filter.  Thus, there
%  are wrap-around effects in the images that are generated using
%  this method.  To generate images that don't suffer from this effect,
%  use MVNLumpyConv
%
%  numimgs is an optional argument to return a number of images.
%  If this option is selected then img is a [numpixels X numimages]
%  matrix.
%
%  Example: img = MVNLumpy(10,zeros(128,128),ones(128,128));
%
%  See Also:  MVNLumpyConv
%

% Get the number of images to generate
if (nargin == 3)
  numimgs = 1;
end

% Get the dimensions of the image
dim = size(muimg);

% Generate the convolution kernel
x = 0:(dim(2)-1);
y = 0:(dim(1)-1);
[X,Y] = meshgrid(x,y);

center = (dim - [1 1])/2;

% The convolution kernel width that gives an AC of a 
% Gaussian w/ lump_simga width is this...
kernel_width = lump_width / sqrt(2);

kernel =  exp(-(1/(2*kernel_width^2))*((X-center(2)).^2 + ...
    (Y-center(1)).^2));

% Properly normalize the kernel such that the diagonal elements
% of A*A' are 1
kernel = kernel / sqrt(sum(kernel(:).*kernel(:)));

if (numimgs == 1)
  % Sample a iid normal with 0 mean and variance of 1
  n = randn(dim);
  % correlate the noise via a convolution
  img = real(ifftshift(ifft2(fft2(fftshift(n)) .* fft2(fftshift(kernel)))));
  % change the variance 
  img = img .* sigimg;
  % change the mean
  img = img + muimg;
else 
  % pre-create the images matrix
  img = zeros(prod(dim),numimgs);
  for i = 1:numimgs,
    % Sample a iid normal with 0 mean and variance of 1
    n = randn(dim);
    % correlate the noise via a convolution
    im = real(ifftshift(ifft2(fft2(fftshift(n)) .* fft2(fftshift(kernel)))));
    % change the variance 
    im = im .* sigimg;
    % change the mean
    im = im + muimg;
    img(:,i) = im(:);
  end
end
