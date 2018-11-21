#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301,W0611,W0102,R0913,R0902
""" Hotelling observer for lumpy mcmc comparison """

import numpy as np
import numpy.random as npr
import scipy.ndimage as snd
from sklearn.metrics import roc_auc_score
from mcmc_lumpy import create_lumpy_background

def main():
    """
        Main
    """
    # Variables
    signal_intensity = 0.1
    var_noise = 0.1
    dim = 64
    gaussian_sigma = 2
    obj_dim1 = [28, 33]
    obj_dim2 = [29, 32]
    num_examples = 100 # for each set

    # Create signal
    signal = np.zeros((64, 64))
    signal[obj_dim1[0]:obj_dim1[1], obj_dim2[0]:obj_dim2[1]] = signal_intensity
    signal[obj_dim2[0]:obj_dim2[1], obj_dim1[0]:obj_dim1[1]] = signal_intensity
    signal = snd.filters.gaussian_filter(signal, gaussian_sigma)

    # make images
    images = []
    backgrounds = np.zeros((dim, dim, num_examples*2))
    for k in range(num_examples):
        print(k)
        # signal present
        b, _, _ = create_lumpy_background()
        noise = npr.normal(0, var_noise**0.5, (dim, dim))
        g = signal + snd.filters.gaussian_filter(b, gaussian_sigma) + noise
        images.append(g)
        backgrounds[:, :, 2*k] = b

        # signal absent images
        b, _, _ = create_lumpy_background()
        noise = npr.normal(0, var_noise**0.5, (dim, dim))
        g = snd.filters.gaussian_filter(b, gaussian_sigma) + noise
        images.append(g)
        backgrounds[:, :, 2*k+1] = b

    # calculate lambdas
    lr = []

    # Apply hotelling observer to data
    N_inv = np.eye(dim**2)*(1/var_noise)
    W = backgrounds - np.repeat(np.mean(backgrounds, axis=2)[:, :, np.newaxis], num_examples*2, axis=2)
    W = np.reshape(W, (dim**2, -1))
    NsNs = np.linalg.inv(np.eye(2*num_examples)+np.matmul(W.transpose(), np.matmul(N_inv, W)))
    K_inv = N_inv - np.matmul(N_inv, np.matmul(W, np.matmul(NsNs, np.matmul(W.transpose(), N_inv))))
    for k, g in enumerate(images):
        print(k)
        lr.append(np.dot(signal.ravel(), np.matmul(K_inv, g.ravel())))

    # Print AUC
    print(roc_auc_score([1, 0]*num_examples, lr))

if __name__ == '__main__':
    main()
