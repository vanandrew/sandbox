#!/usr/bin/env python3
# pylint: disable=R0914
"""
A CNN emulating a hotelling observer
"""

import shutil
import tensorflow as tf
import numpy as np
from lumpybg import *
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

def create_tf_graph(layers=1):
    """
        Creates the tensorflow graph
    """

    # setup placeholder for network input
    net_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])

    # setup layers
    conv_stack, _ = setup_conv_layers(layers, net_input)
    pool0 = tf.layers.max_pooling2d(inputs=conv_stack, pool_size=(2, 2), strides=2, name='pool0')
    dense0 = tf.layers.dense(
        inputs=tf.reshape(pool0, [-1, int(pool0.shape[1]*pool0.shape[2]*32)]),
        units=512, activation=tf.nn.relu,
        name='dense0')
    readout = tf.squeeze(tf.layers.dense(inputs=dense0, units=1))

    # create a placeholder to feed in data for loss function
    label_cmp = tf.placeholder(tf.bool)

    # set loss function
    loss = tf.losses.sigmoid_cross_entropy(label_cmp, readout)

    # setup optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss)

    # return all outputs
    return net_input, label_cmp, readout, loss, train_op

def create_metrics(loss, readout, label_cmp):
    """
    Metrics for examining network performance
    """

    # create loss summary
    loss_summary = tf.summary.scalar('loss', loss)

    # check AUC on training set
    train_auc, train_auc_op = tf.metrics.auc(label_cmp, tf.sigmoid(readout), name='train_auc')
    train_summary = tf.summary.scalar('training_auc', train_auc)

    # check AUC on validation set
    val_auc, val_auc_op = tf.metrics.auc(label_cmp, tf.sigmoid(readout), name='val_auc')
    val_summary = tf.summary.scalar('validation_auc', val_auc)

    # create summary op
    train_summary_op = tf.summary.merge([loss_summary, train_summary])
    val_summary_op = tf.summary.merge([val_summary])

    return train_auc_op, val_auc_op, train_summary_op, val_summary_op

def main():
    """
    Main Function
    """

    # get the lumpy background data
    train_set, val_set, train_label, val_label, _, _ = data_import('dataset20.mat',48000)

    # create graph
    net_input, label_cmp, readout, loss, train_op = create_tf_graph()

    # define metrics
    train_auc_op, val_auc_op, train_summary_op, val_summary_op = create_metrics(
        loss,
        readout,
        label_cmp)

    # get running vars so we can reset them
    train_auc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="training_auc")
    val_auc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="validation_auc")

    # create operations to reset variables
    train_auc_vars_initializer = tf.variables_initializer(var_list=train_auc_vars)
    val_auc_vars_initializer = tf.variables_initializer(var_list=val_auc_vars)

    # save model
    saver = tf.train.Saver(max_to_keep=10)

    # create a variable to save the latest validation AUC
    best_validation_auc = 0

    # setup and run tensorflow session
    with tf.Session() as sess:
        # initialize all variables
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # setup writer for tensorbaord log files
        shutil.rmtree('./logdir', ignore_errors=True) # remove any existing log dirs
        writer = tf.summary.FileWriter('./logdir', sess.graph)

        # run epochs
        for epoch in range(100000):
            # get mini batch for training
            tset, lset = get_batch(2048, train_set, train_label)

            # train on batch
            sess.run(train_op, feed_dict={net_input: tset, label_cmp: lset})

            if (epoch + 1) % 10 == 0:
                # reset all summary values
                sess.run(train_auc_vars_initializer)
                sess.run(val_auc_vars_initializer)

                # calculate training stats
                train_loss, train_auc_val = sess.run(
                    [loss, train_auc_op],
                    feed_dict={net_input: tset, label_cmp: lset})

                # get auc on validation set
                val_auc_val = sess.run(
                    val_auc_op,
                    feed_dict={net_input: val_set, label_cmp: val_label})

                # log summaries
                train_summary_out = sess.run(
                    train_summary_op,
                    feed_dict={net_input: tset, label_cmp: lset})
                val_summary_out = sess.run(val_summary_op)
                writer.add_summary(train_summary_out, epoch+1)
                writer.add_summary(val_summary_out, epoch+1)

                # output logging info
                tf.logging.info(("Epoch: {}, Training Loss: {}, Training AUC: {},"
                                 " Validation AUC: {}").format(
                                     epoch+1,
                                     train_loss,
                                     train_auc_val,
                                     val_auc_val))

                # save model checkpoint if best
                if best_validation_auc < val_auc_val:
                #if True:
                    save_path = saver.save(
                        sess,
                        './saved_models_lb/ho_cnn_model.ckpt',
                        global_step=epoch+1)
                    tf.logging.info(("Model saved at {}".format(save_path)))
                    # set best_validation_auc to current
                    best_validation_auc = val_auc_val

if __name__ == '__main__':
    main()
