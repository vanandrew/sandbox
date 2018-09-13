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

def main():
    """
    Main Function
    """

    # get the lumpy background data
    train_set, val_set, train_label, val_label = data_import('dataset.mat', 9900)

    # setup placeholder for network input
    net_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])

    # setup layers
    conv0 = tf.layers.conv2d(
        inputs=net_input,
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        name='conv0')
    pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=(2, 2), strides=2, name='pool0')
    dense0 = tf.layers.dense(
        inputs=tf.reshape(pool0, [-1, int(pool0.shape[1]*pool0.shape[2]*32)]),
        units=1024, activation=tf.nn.relu,
        name='dense0')
    readout = tf.squeeze(tf.layers.dense(inputs=dense0, units=1))

    # create a placeholder to feed in data for loss function
    label_cmp = tf.placeholder(tf.float32)

    # set loss function
    loss = tf.losses.sigmoid_cross_entropy(label_cmp, readout)
    loss_summary = tf.summary.scalar('loss', loss)

    # setup optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # check accuracy on training set
    training_accuracy, ta_op = tf.metrics.accuracy(label_cmp, tf.sigmoid(readout))
    train_summary = tf.summary.scalar('training_accuracy', training_accuracy)

    # check accuracy on validation set
    validation_accuracy, val_op = tf.metrics.accuracy(val_label, tf.sigmoid(readout))
    validation_summary = tf.summary.scalar('validation_accuracy', validation_accuracy)

    # check AUC on validation set
    auc, auc_op = tf.metrics.auc(val_label, tf.sigmoid(readout))
    auc_summary = tf.summary.scalar('AUC', auc)

    # create summary op
    summary_op = tf.summary.merge([loss_summary, train_summary])
    val_summary_op = tf.summary.merge([validation_summary, auc_summary])

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

            # train on batch
            summary, training_accuracy_value, _, train_loss = sess.run(
                [summary_op, ta_op, train_op, loss],
                feed_dict={net_input: tset, label_cmp: lset})
            writer.add_summary(summary, epoch)

            # get auc on validation set
            val_summary_out, auc_val, validation_accuracy_value, val_loss = sess.run(
                [val_summary_op, auc_op, val_op, loss],
                feed_dict={net_input: val_set, label_cmp: val_label})
            writer.add_summary(val_summary_out, epoch)

            # output logging info
            tf.logging.info(("Epoch: {}, Training Loss: {}, Training Accuracy: {},"
                             " Validation Loss: {}, Validation Accuracy: {}, AUC {}").format(
                                 epoch,
                                 train_loss,
                                 training_accuracy_value,
                                 val_loss,
                                 validation_accuracy_value,
                                 auc_val))

if __name__ == '__main__':
    main()
