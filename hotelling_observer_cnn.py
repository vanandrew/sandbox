#!/usr/bin/env python3
# pylint: disable=R0914
"""
A CNN emulating a hotelling observer
"""

import shutil
import tensorflow as tf
import numpy as np
from lumpybg import data_import
tf.logging.set_verbosity(tf.logging.INFO)

def get_batch(n_size, data, labels):
    """
    Return n-size batch of data and labels respectively
    """
    # data and labels must be same length in 1st dim
    assert len(data) == len(labels), "Data and Labels not the same length!"

    # get indicies for shuffle
    idx = np.arange(len(data))
    np.random.shuffle(idx)

    # return batch
    return data[idx[0:n_size]], labels[idx[0:n_size]]

def setup_conv_layers(num_layers, net_input):
    """
    sets up n number convolutional layers
    """

    # num_layers must be greater than 0
    assert num_layers > 0, "num_layers must be > 0!"

    # setup initial input to conv layer
    conv_input = net_input

    # create list of layers
    conv_layers = []

    # append conv layer to list for each num in num_layers
    for num in range(num_layers):
        conv_layers.append(
            tf.layers.conv2d(
                inputs=conv_input,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu,
                name='conv{}'.format(num)))
        # assign the latest layer to conv_input
        conv_input = conv_layers[-1]

    # return the latest layer and the list of conv layers
    return conv_input, conv_layers

def main():
    """
    Main Function
    """

    # get the lumpy background data
    train_set, val_set, train_label, val_label = data_import('simple_dataset.mat', 9900)

    # setup placeholder for network input
    net_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])

    # setup layers
    conv_stack, _ = setup_conv_layers(3, net_input)
    pool0 = tf.layers.max_pooling2d(inputs=conv_stack, pool_size=(2, 2), strides=2, name='pool0')
    dense0 = tf.layers.dense(
        inputs=tf.reshape(pool0, [-1, int(pool0.shape[1]*pool0.shape[2]*32)]),
        units=8096, activation=tf.nn.relu,
        name='dense0')
    readout = tf.squeeze(tf.layers.dense(inputs=dense0, units=1))

    # get an the final activation layer for visualization and reshape
    activation = tf.transpose(conv_stack, [3, 1, 2, 0])
    activation_layer_abs = tf.summary.image('signal_abs', activation, max_outputs=32)
    activation_layer_pres = tf.summary.image('signal_pres', activation, max_outputs=32)

    # create a placeholder to feed in data for loss function
    label_cmp = tf.placeholder(tf.bool)

    # set loss function
    loss = tf.losses.sigmoid_cross_entropy(label_cmp, readout)
    loss_summary = tf.summary.scalar('loss', loss)

    # setup optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # check accuracy on training set
    train_pred = tf.greater_equal(tf.sigmoid(readout), 0.5)
    training_accuracy, ta_op = tf.metrics.accuracy(label_cmp, train_pred, name='train_acc')
    train_summary = tf.summary.scalar('training_accuracy', training_accuracy)

    # check accuracy on validation set
    val_pred = tf.greater_equal(tf.sigmoid(readout), 0.5)
    validation_accuracy, val_op = tf.metrics.accuracy(
        val_label, val_pred, name='val_acc')
    validation_summary = tf.summary.scalar('validation_accuracy', validation_accuracy)

    # check AUC on validation set
    auc, auc_op = tf.metrics.auc(val_label, tf.sigmoid(readout), name='auc_met')
    auc_summary = tf.summary.scalar('AUC', auc)

    # create summary op
    train_summary_op = tf.summary.merge([loss_summary, train_summary])
    val_summary_op = tf.summary.merge([validation_summary, auc_summary])

    # get running vars so we can reset them
    train_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_acc")
    val_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="val_acc")
    auc_met_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="auc_met")

    # create operations to reset variables
    train_acc_vars_initializer = tf.variables_initializer(var_list=train_acc_vars)
    val_acc_vars_initializer = tf.variables_initializer(var_list=val_acc_vars)
    auc_met_vars_initializer = tf.variables_initializer(var_list=auc_met_vars)

    # setup and run tensorflow session
    with tf.Session() as sess:
        # initialize all variables
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # setup writer for tensorbaord log files
        shutil.rmtree('./logdir', ignore_errors=True) # remove any existing log dirs
        writer = tf.summary.FileWriter('./logdir', sess.graph)

        # run epochs
        for epoch in range(100):
            # get mini batch for training
            tset, lset = get_batch(32, train_set, train_label)

            # reset all summary values
            sess.run(train_acc_vars_initializer)
            sess.run(val_acc_vars_initializer)
            sess.run(auc_met_vars_initializer)

            # train on batch
            sess.run(train_op, feed_dict={net_input: tset, label_cmp: lset})

            # calculate training stats
            training_accuracy_value, train_loss = sess.run(
                [ta_op, loss],
                feed_dict={net_input: tset, label_cmp: lset})

            # get auc on validation set
            auc_val, validation_accuracy_value = sess.run(
                [auc_op, val_op],
                feed_dict={net_input: val_set})

            # log summaries
            train_summary_out = sess.run(
                train_summary_op,
                feed_dict={net_input: tset, label_cmp: lset})
            val_summary_out = sess.run(val_summary_op)
            writer.add_summary(train_summary_out, epoch)
            writer.add_summary(val_summary_out, epoch)

            # get the activation layer and pass a signal_absent/signal_present example
            sig_abs_out = sess.run(
                activation_layer_abs,
                feed_dict={net_input: np.expand_dims(val_set[0, :, :, :], axis=0)})
            sig_pres_out = sess.run(
                activation_layer_pres,
                feed_dict={net_input: np.expand_dims(val_set[val_set.shape[0]//2, :, :, :], axis=0)})
            writer.add_summary(sig_abs_out, epoch)
            writer.add_summary(sig_pres_out, epoch)

            # output logging info
            tf.logging.info(("Epoch: {}, Training Loss: {}, Training Accuracy: {},"
                             " Validation Accuracy: {}, AUC {}").format(
                                 epoch,
                                 train_loss,
                                 training_accuracy_value,
                                 validation_accuracy_value,
                                 auc_val))

if __name__ == '__main__':
    main()
