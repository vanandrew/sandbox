"""
    Imports lumpy background data from .mat file
"""
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import normalize

def data_import(matfile, train_size):
    """
        Imports the lumpy background data from the mat file. It
        expects the matfile to contain the signal_absent and signal_present
        variables, which are d1 x d2 x n arrays (where d1 and d2 are
        dimensions in the array, and n is the number of samples).

        The train_size parameter sets the boundary for dividing the data
        into a training and validation set. Setting train_size to m, will
        use m examples (from present/absent each) for the training set, and
        the rest for validation.

        Data is normalized.

        Data is reshaped into (batch_size,image_dimension_1,image_dimension_2,1)

        This function also creates the labels for each set, where 1 is equal to
        signal present and 0 is equal to signal absent.

        This function returns 4 outputs: training_set, validation_set,
        training_labels, and validation_labels respectively as numpy arrays.
    """

    # load dataset
    mat_contents = sio.loadmat(matfile)
    signal_absent = mat_contents['signal_absent']
    signal_present = mat_contents['signal_present']

    # format data for tensorflow
    signal_absent = np.swapaxes(signal_absent, 1, 2)
    signal_absent = np.swapaxes(signal_absent, 0, 1)
    signal_present = np.swapaxes(signal_present, 1, 2)
    signal_present = np.swapaxes(signal_present, 0, 1)

    # split the dataset into train,validation
    signal_absent_train = signal_absent[0:train_size, :, :]
    signal_present_train = signal_present[0:train_size, :, :]
    signal_absent_val = signal_absent[train_size:, :, :]
    signal_present_val = signal_present[train_size:, :, :]

    # now combine the positive/negative examples into one array
    train_set = np.concatenate((signal_absent_train, signal_present_train), axis=0)
    val_set = np.concatenate((signal_absent_val, signal_present_val), axis=0)

    # reshape for normalization
    #train_set = np.reshape(train_set, (-1, 64*64))
    #val_set = np.reshape(val_set, (-1, 64*64))

    # normalize images
    #train_set = normalize(train_set)
    #val_set = normalize(val_set)

    # now reshape for features layer
    train_set = np.reshape(train_set, (-1, 64, 64, 1))
    val_set = np.reshape(val_set, (-1, 64, 64, 1))

    # create labels for training and validation set
    labels_train = np.concatenate([
        np.zeros(signal_absent_train.shape[0], dtype=bool),
        np.ones(signal_present_train.shape[0], dtype=bool)
    ], axis=0)
    labels_val = np.concatenate([
        np.zeros(signal_absent_val.shape[0], dtype=bool),
        np.ones(signal_present_val.shape[0], dtype=bool)
    ], axis=0)

    # return sets
    return train_set, val_set, labels_train, labels_val
