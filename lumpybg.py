"""
    Imports lumpy background data from .mat file
"""
# pylint: disable=W0102,R0913,R0914
import scipy.io as sio
import scipy.ndimage as snd
import numpy as np

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

        Data is min-max normalized.

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

    # normalize images
    tmin = train_set.flatten().min()
    tmax = train_set.flatten().max()
    train_set = (train_set - tmin)/(tmax - tmin)
    val_set = (val_set - tmin)/(tmax - tmin)

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
    return train_set, val_set, labels_train, labels_val, tmax, tmin

def ske_bke_import(signal_intensity=0.1, background_intensity=20, mean_noise=0,
                   var_present_noise=0.01, var_absent_noise=0.05, gaussian_sigma=0.5,
                   num_images=10000, image_size=64, obj_dim1=[28, 33], obj_dim2=[29, 32],
                   train_idx=9000, val_idx=10000):
    """
        Creates SKE/BKE data for classification
    """

    # Create list to store noise images
    noise_present = []
    noise_absent = []

    # Create noise images
    for _ in range(num_images):
        # Create measurement noise
        noise_present.append(
            np.random.normal(mean_noise, var_present_noise**0.5, (image_size, image_size)))
        noise_absent.append(
            np.random.normal(mean_noise, var_absent_noise**0.5, (image_size, image_size)))

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

    # stack data
    train_set = np.stack(train_signal_absent + train_signal_present, axis=0)
    val_set = np.stack(val_signal_absent + val_signal_present, axis=0)

    # create labels for training and validation set
    labels_train = np.concatenate([
        np.zeros(len(train_signal_absent), dtype=bool),
        np.ones(len(train_signal_present), dtype=bool)
    ], axis=0)
    labels_val = np.concatenate([
        np.zeros(len(val_signal_absent), dtype=bool),
        np.ones(len(val_signal_present), dtype=bool)
    ], axis=0)

    # normalize images
    tmin = train_set.flatten().min()
    tmax = train_set.flatten().max()
    train_set = (train_set - tmin)/(tmax - tmin)
    val_set = (val_set - tmin)/(tmax - tmin)

    # now reshape for features layer
    train_set = np.reshape(train_set, (-1, 64, 64, 1))
    val_set = np.reshape(val_set, (-1, 64, 64, 1))

    # return sets
    return train_set, val_set, labels_train, labels_val, tmax, tmin
