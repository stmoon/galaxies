"""
made by STMOON and YJCHOI
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import prepare_data
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


prepare_data.prepre_resize_train()      ## Create Train Data
prepare_data.prepare_resize_test()      ## Create Train Data
prepare_data.prepare_csv()


### Hyper parameter
IMAGE_WIDTH =  128
IMAGE_HEIGHT = 128
KEEP_PROB = 0.7
TRAIN_EPOCH = 30
BATCH_SIZE = 10#50
NUM_TOTAL_TRAINING_DATA = 1000
NUM_THREADS = 2
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 3
FILTER_SIZE = 3
POOLING_SIZE = 2


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w = tf.get_variable(name+"_w", shape=[FILTER_SIZE, FILTER_SIZE, size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    h1 = conv + b
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=True, scope=name)
    act = tf.nn.relu(h2, 'relu')
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, is_relu=True, name="fc"):
  with tf.name_scope(name):
    flat_input = tf.reshape(input, [-1, size_in])
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w = tf.get_variable(name+"_w", shape=[size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    h1  = tf.matmul(flat_input, w) + b
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=True, scope=name)
    if is_relu :
	act = tf.nn.relu(h2, 'relu')
    else :
	act = h2
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def model(learning_rate, hparam) :

    # open session
    tf.reset_default_graph()
    sess = tf.Session()

    # input your path
    csv_file =  tf.train.string_input_producer([prepare_data.MODIFIED_TRAIN_LABEL_CSV_PATH], name='filename_queue', shuffle=True)       
    csv_reader = tf.TextLineReader()
    _,line = csv_reader.read(csv_file)


    record_defaults = [[""]] + [[1.]] * 37 
    rows = tf.decode_csv(line,record_defaults=record_defaults)
    label_decoded = tf.stack([rows[1:NUM_CLASSES+1]])

    image_decoded = tf.image.decode_jpeg(tf.read_file(rows[0]),channels=3)

    image_cast = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])


    # similary tf.placeholder
    # Training batch set
    image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='X')
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='Y')


    ### Graph 
    conv1 = conv_layer(X, 3, 32, name='conv1')
    conv2 = conv_layer(conv1, 32, 64, name='conv2')
    conv3 = conv_layer(conv2, 64, 128, name='conv3')
    conv4 = conv_layer(conv3, 128, 256, name='conv4')
    conv5 = conv_layer(conv4, 256, 512, name='conv5')
    conv6 = conv_layer(conv5, 512, 1024, name='conv6')
    conv7 = conv_layer(conv6, 1024, 2048, name='conv7')
    fc1   = fc_layer(conv7, 2048, 2048, name='fc1')
    fc2   = fc_layer(fc1, 2048, NUM_CLASSES, is_relu=False, name='fc2')
    logits = tf.nn.softmax(fc2, name=None)

    with tf.name_scope('train') :
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) 
	cost_hist = tf.summary.scalar('cost', cost) 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('distance') :
	dist = tf.reduce_mean(tf.square(tf.subtract(logits, Y)), axis=1)
	dist = tf.reduce_mean(tf.sqrt(dist))
	dist_hist = tf.summary.scalar('distance', dist)


    with tf.name_scope('accuracy') :
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	acc_hist = tf.summary.scalar('accuracy', accuracy)

    summary  = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/'+hparam)
    writer.add_graph(sess.graph)


    ## RUN
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(TRAIN_EPOCH):
	total_batch = int(NUM_TOTAL_TRAINING_DATA/BATCH_SIZE)
	for i in range(total_batch):
	    batch_x, batch_y = sess.run([image_batch, label_batch])
	    batch_y = batch_y.reshape(BATCH_SIZE, NUM_CLASSES)
	    opt, summ, acc = sess.run([optimizer, summary, accuracy], feed_dict={X: batch_x, Y: batch_y})

	writer.add_summary(summ, epoch)
	print "epoch[%d]: acc(%f) " % (epoch, acc) 

       
	    
    coord.request_stop()
    coord.join(threads) 

def main() :
    count = 0
    for learning_rate in [ 1.0*1E-5  ] :
	model(learning_rate, "param%d_%f" % (count, learning_rate))
	count += 1

if __name__ == '__main__':
    main()
