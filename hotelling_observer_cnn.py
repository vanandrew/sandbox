#!/usr/bin/env python3

import tensorflow as tf
import scipy.io as sio
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)

# settings
train_size = 1000

# load dataset
mat_contents = sio.loadmat('dataset.mat')
signal_absent = mat_contents['signal_absent']
signal_present = mat_contents['signal_present']

# format data for tensorflow
signal_absent = np.swapaxes(signal_absent,1,2)
signal_absent = np.swapaxes(signal_absent,0,1)
signal_present = np.swapaxes(signal_present,1,2)
signal_present = np.swapaxes(signal_present,0,1)

# split the dataset into train,validation
signal_absent_train = signal_absent[0:train_size,:,:]
signal_present_train = signal_present[0:train_size,:,:]
signal_absent_val = signal_absent[train_size:1100,:,:,]
signal_present_val = signal_present[train_size:1100,:,:]

# now combine the positive/negative examples into one array
train_set = np.concatenate((signal_absent_train,signal_present_train),axis=0)
val_set = np.concatenate((signal_absent_val,signal_present_val),axis=0)

# now reshape for features layer
train_set = np.reshape(train_set,(-1,64,64,1))
val_set = np.reshape(val_set,(-1,64,64,1))

# create labels for training and validation set
labels_train = tf.concat([
        tf.zeros(signal_absent_train.shape[0],dtype=tf.float32),
        tf.ones(signal_present_train.shape[0],dtype=tf.float32)
    ],axis=0)
labels_val = tf.concat([
        tf.zeros(signal_absent_val.shape[0],dtype=tf.float32),
        tf.ones(signal_present_val.shape[0],dtype=tf.float32)
    ],axis=0)

# setup placeholder for network input
net_input = tf.placeholder(tf.float32,shape=[None,64,64,1])

# setup layers
conv0 = tf.layers.conv2d(inputs=net_input,filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu,name='conv0')
pool0 = tf.layers.max_pooling2d(inputs=conv0,pool_size=(2,2),strides=2,name='pool0')
dense0 = tf.layers.dense(inputs=tf.reshape(pool0,[-1,int(pool0.shape[1]*pool0.shape[2]*32)]),units=1024,activation=tf.nn.relu,name='dense0')
readout = tf.squeeze(tf.layers.dense(inputs=dense0,units=1))

# set loss function
loss = tf.losses.sigmoid_cross_entropy(readout,labels_train)
loss_summary = tf.summary.scalar('loss',loss)

# setup optimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

# check accuracy on training set
training_accuracy,ta_op = tf.metrics.accuracy(labels_train,tf.sigmoid(readout))
train_summary = tf.summary.scalar('training_accuracy',training_accuracy)

# check AUC on validation set
AUC,AUC_op = tf.metrics.auc(labels_val,tf.sigmoid(readout))
AUC_summary = tf.summary.scalar('AUC',AUC)

# create summary op
summary_op = tf.summary.merge([loss_summary,train_summary])

# setup and run tensorflow session
with tf.Session() as sess:
    # initialize all variables
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    writer = tf.summary.FileWriter('./logdir',sess.graph)
    for i in range(1000):
        summary,_,_ = sess.run([summary_op, ta_op, train_op], feed_dict={net_input: train_set})
        writer.add_summary(summary,i)
        AUC_summary_out,_ = sess.run([AUC_summary, AUC_op], feed_dict={net_input: val_set})
        writer.add_summary(AUC_summary_out,i)
