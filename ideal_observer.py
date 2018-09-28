#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301
"""
    Ideal observer refactored
"""

import scipy.ndimage as snd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from hotelling_observer_cnn import create_tf_graph

# Settings
def main():
    """
    Generate samples
    """
    signal_intensity = 20
    background_intensity = 100
    var_present_noise = 1500
    var_absent_noise = 1600
    gaussian_sigma = 0.5
    image_size = 64
    obj_dim1 = [30, 34]
    obj_dim2 = [31, 33]
    num_images = 4400
    train_idx = 4200
    val_idx = 4400

    # Create list to store noise images
    noise_present = []
    noise_absent = []

    # Create noise images
    for _ in range(num_images):
        # Create measurement noise
        noise_present.append(np.random.normal(0, var_present_noise**0.5, (image_size, image_size)))
        noise_absent.append(np.random.normal(0, var_absent_noise**0.5, (image_size, image_size)))

    # Create background
    background = np.ones((image_size, image_size))*background_intensity

    # Create signal
    signal = np.zeros((image_size, image_size))
    signal[obj_dim1[0]:obj_dim1[1], obj_dim2[0]:obj_dim2[1]] = signal_intensity
    signal[obj_dim2[0]:obj_dim2[1], obj_dim1[0]:obj_dim1[1]] = signal_intensity

    # Create signal absent/present images
    background_gauss = snd.filters.gaussian_filter(background, gaussian_sigma)
    signal_gauss = snd.filters.gaussian_filter(signal+background, gaussian_sigma)
    signal_absent = [background_gauss+nse for nse in noise_absent]
    signal_present = [signal_gauss+nse for nse in noise_present]

    # split train/val set
    val_signal_absent = signal_absent[train_idx:val_idx]
    val_signal_present = signal_present[train_idx:val_idx]

    # flatten arrays
    val_signal_absent = np.transpose(np.vstack([n.flatten() for n in val_signal_absent]))
    val_signal_present = np.transpose(np.vstack([n.flatten() for n in val_signal_present]))

    # combine validation images
    data_array = np.hstack((val_signal_absent, val_signal_present))

    # calculate difference of 2 classes
    avg_t = np.reshape(signal_gauss-background_gauss, (1, image_size**2))

    # calculate test statistic
    l_pw = np.transpose(np.matmul(avg_t, data_array))

    # calculate nonlinear test statistic
    s2 = np.reshape(signal_gauss, (image_size**2, 1))
    s1 = np.reshape(background_gauss, (image_size**2, 1))
    t1 = np.reshape(np.diagonal(np.matmul(np.transpose(data_array), data_array))*(var_present_noise - var_absent_noise), (-1, 1))
    t2 = 2*np.matmul(np.transpose(data_array), (s2*var_absent_noise - s1*var_present_noise))
    l_nonlin = t1 + t2

    # format validation images for cnn
    tmin = data_array.flatten().min()
    tmax = data_array.flatten().max()
    normal = (data_array - tmin)/(tmax - tmin)
    cnn_data_array = np.reshape(np.transpose(normal), (-1, image_size**2, image_size**2, 1))

    # load up ho cnn
    net_input, _, readout, _, _ = create_tf_graph()
    sess = tf.Session()
    tf.train.Saver().restore(sess, './saved_models/ho_cnn_model.ckpt-24290')

    # pass val input
    readout_output = sess.run(readout, feed_dict={net_input: cnn_data_array})

    # print performance
    img_cls = np.array([0]*(val_idx-train_idx) + [1]*(val_idx-train_idx))
    [fpr, tpr, _] = roc_curve(img_cls, l_pw)
    [fpr_nl, tpr_nl, _] = roc_curve(img_cls, l_nonlin)
    [fpr_cnn, tpr_cnn, _] = roc_curve(img_cls, readout_output)
    print("AUC: {}".format(roc_auc_score(img_cls, l_pw)))
    print("nonlin AUC: {}".format(roc_auc_score(img_cls, l_nonlin)))
    print("CNN AUC: {}".format(roc_auc_score(img_cls, readout_output)))
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.plot(fpr_nl, tpr_nl)
    plt.plot(fpr_cnn, tpr_cnn)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if __name__ == '__main__':
    main()
