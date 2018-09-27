#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915
"""
    Ideal observer refactored
"""

from functools import reduce
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
    var_absent_noise = 2000
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
    train_signal_absent = signal_absent[0:train_idx]
    train_signal_present = signal_present[0:train_idx]
    val_signal_absent = signal_absent[train_idx:val_idx]
    val_signal_present = signal_present[train_idx:val_idx]

    # Generate average images for signal present/signal absent
    avg_signal_absent = reduce(lambda x, y: x+y, train_signal_absent)/4200
    avg_signal_present = reduce(lambda x, y: x+y, train_signal_present)/4200

    # flatten arrays
    avg_signal_absent_array = avg_signal_absent.flatten()
    avg_signal_present_array = avg_signal_present.flatten()
    val_signal_absent = np.transpose(np.vstack([n.flatten() for n in val_signal_absent]))
    val_signal_present = np.transpose(np.vstack([n.flatten() for n in val_signal_present]))

    # combine validation images
    data_array = np.hstack((val_signal_absent, val_signal_present))

    # calculate difference of avg 2 classes
    avg_t = avg_signal_present_array-avg_signal_absent_array

    # calculate test statistic
    l_pw = np.matmul(avg_t, data_array)

    # format validation images for cnn
    tmin = data_array.flatten().min()
    tmax = data_array.flatten().max()
    normal = (data_array - tmin)/(tmax - tmin)
    cnn_data_array = np.reshape(np.transpose(normal), (-1, 64, 64, 1))

    # load up ho cnn
    net_input, _, readout, _, _ = create_tf_graph()
    sess = tf.Session()
    tf.train.Saver().restore(sess, './saved_models/ho_cnn_model.ckpt-24290')

    # pass val input
    readout_output = sess.run(readout, feed_dict={net_input: cnn_data_array})

    # print performance
    img_cls = np.array([0]*(val_idx-train_idx) + [1]*(val_idx-train_idx))
    [fpr, tpr, _] = roc_curve(img_cls, l_pw)
    [fpr_cnn, tpr_cnn, _] = roc_curve(img_cls, readout_output)
    print("AUC: {}".format(roc_auc_score(img_cls, l_pw)))
    print("CNN AUC: {}".format(roc_auc_score(img_cls, readout_output)))
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.plot(fpr_cnn,tpr_cnn)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if __name__ == '__main__':
    main()
