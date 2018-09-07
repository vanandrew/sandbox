clc;
clear;
close all;

% add image quality scripts to path
addpath('ImageQuality');

% settings
signal_intensity = 0.1;
signal_sigma = 2;
bg_offset = 20;
bg_sigma = 10;
noise_var = 0.01;
num_images = 2500;
model = 2400;
img_dim = 64;
sig_dim1 = 29:33;
sig_dim2 = 30:32;

% convenience functions
flatten = @(img) reshape(img,img_dim^2,num_images);

% generate the signal image
signal = zeros(img_dim,img_dim);
signal(sig_dim1,sig_dim2) = signal_intensity;
signal(sig_dim2,sig_dim1) = signal_intensity;

% generate images
noise = zeros(img_dim,img_dim,num_images);
signal_absent = zeros(img_dim,img_dim,num_images);
signal_present = zeros(img_dim,img_dim,num_images);
bg = zeros(img_dim,img_dim,num_images,2);
for n=1:num_images
    % generate random noise
    noise(:,:,n) = normrnd(0,sqrt(noise_var),[img_dim,img_dim]);
    
    % background
    bg(:,:,n,1) = LumpyBgnd([img_dim,img_dim],25,bg_offset,'GaussLmp',[1,bg_sigma]);
    bg(:,:,n,2) = LumpyBgnd([img_dim,img_dim],25,bg_offset,'GaussLmp',[1,bg_sigma]);
    
    % generate signal-absent images
    signal_absent(:,:,n) = imgaussfilt(bg(:,:,n,1),signal_sigma)+noise(:,:,n);
    
    % generate signal-present images
    signal_present(:,:,n) = imgaussfilt(bg(:,:,n,2),signal_sigma)+noise(:,:,n);
end
    
% display examples
% figure, colormap gray, imagesc(signal_absent(:,:,1));
% figure, colormap gray, imagesc(signal_present(:,:,1));

% flatten signal-present/signal-absent images/background images
signal_absent_array = flatten(signal_absent);
signal_present_array = flatten(signal_present);
bg_absent_array = flatten(bg(:,:,:,1));
bg_present_array = flatten(bg(:,:,:,2));

% split into model set and validation set
sa_model_array = signal_absent_array(:,1:model);
sp_model_array = signal_present_array(:,1:model);
bg_model_array = [bg_absent_array(:,1:model), bg_present_array(:,1:model)];
sa_val_array = signal_absent_array(:,model+1:num_images);
sp_val_array = signal_present_array(:,model+1:num_images);

% find average difference in present vs absent
avg_diff = mean(sp_model_array,2) - mean(sa_model_array,2);
% figure, colormap gray, imagesc(reshape(avg_diff,img_dim,img_dim),[]);

% calculate inverse covariance matrix of background
cov_noise = eye(img_dim^2)*noise_var;
inv_cov_noise = inv(cov_noise);
W = bg_model_array - mean(bg_model_array,2);
NsNs = inv(eye(2*model)+(W')*inv_cov_noise*W); %#ok<*MINV>
inv_cov = inv_cov_noise - inv_cov_noise*W*NsNs*(W')*inv_cov_noise;

% calculate the inverse covariance matrix for the background

% calculate test statistic for the hotelling observer
lambda_Hot = (avg_diff)'*inv_cov*[sa_val_array,sp_val_array];

% generate ROC curve
[x,y,T,AUC] = perfcurve([zeros(1,100),ones(1,100)],lambda_Hot,1);
figure, plot(x,y), xlabel('False Positive Rate'), ylabel('True Positive Rate');
disp(['AUC: ', num2str(AUC)]);
