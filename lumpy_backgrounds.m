clc;
clear;
close all;

% add image quality scripts to path
addpath('ImageQuality');

% settings
signal_intensity = 100;
signal_sigma = 2;
bg_offset = 100;
noise_var = 110;
num_images = 10;
img_dim = 64;
sig_dim1 = 29:33;
sig_dim2 = 30:32;

% generate the signal image
signal = zeros(img_dim,img_dim);
signal(sig_dim1,sig_dim2) = signal_intensity;
signal(sig_dim2,sig_dim1) = signal_intensity;
signal = imgaussfilt(signal,signal_sigma);

% generate images
background = zeros(img_dim,img_dim,num_images);
noise = zeros(img_dim,img_dim,num_images);
signal_absent = zeros(img_dim,img_dim,num_images);
signal_present = zeros(img_dim,img_dim,num_images);
for n=1:num_images
    % generate Lumpy Backgrounds
    background(:,:,n) = LumpyBgnd([img_dim,img_dim],25,bg_offset,'GaussLmp',[1,10]);
    
    % generate random noise
    noise(:,:,n) = normrnd(0,sqrt(noise_var),[img_dim,img_dim]);
    
    % generate signal-absent images
    signal_absent = background(:,:,n)+noise(:,:,n);
    
    % generate signal-present images
    signal_present = signal+background(:,:,n)+noise(:,:,n);
end

% display examples
figure, colormap gray, imagesc(signal_absent(:,:,1));
figure, colormap gray, imagesc(signal_present(:,:,1));