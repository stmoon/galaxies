"""
made by STMOON and YJCHOI
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import prepare_data
import csv


prepare_data.prepre_resize_train()      ## Create Train Data
prepare_data.prepare_resize_test()      ## Create Train Data
prepare_data.prepare_csv()


### Hyper parameter
IMAGE_WIDTH =  128
IMAGE_HEIGHT = 128
KEEP_PROB = 0.7
LEARNING_RATE = 1e-3
TRAIN_EPOCH = 100
BATCH_SIZE = 50
NUM_TOTAL_TRAINING_DATA = 1000
NUM_THREADS = 4
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 3
FILTER_SIZE = 2
POOLING_SIZE = 2


def conv_layer(input, size_in, size_out, name="conv", stddev=0.01):
  with tf.name_scope(name):
    w_init = tf.contrib.layers.xavier_initializer_conv2d()
    w = tf.get_variable(name+"_w", shape=[FILTER_SIZE, FILTER_SIZE, size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram(name+"_weights", w)
    tf.summary.histogram(name+"_biases", b)
    tf.summary.histogram(name+"_activations", act)
    return tf.nn.max_pool(act, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc", stddev=0.01):
  with tf.name_scope(name):
    flat_input = tf.reshape(input, [-1, size_in])
    w_init = tf.contrib.layers.xavier_initializer();
    w = tf.get_variable(name+"_w", shape=[size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(flat_input, w) + b)
    tf.summary.histogram(name+"_weights", w)
    tf.summary.histogram(name+"_biases", b)
    tf.summary.histogram(name+"_activations", act)
    return act


def model(learning_rate, std_dev, hparam) :

    # open session
    tf.reset_default_graph()
    sess = tf.Session()

    # input your path
    csv_file =  tf.train.string_input_producer([prepare_data.MODIFIED_TRAIN_LABEL_CSV_PATH], name='filename_queue', shuffle=True)       
    csv_reader = tf.TextLineReader()
    _,line = csv_reader.read(csv_file)

    record_defaults = [[""], 
		       [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], 
		       [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], 
		       [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], 
		       [1.], [1.], [1.], [1.], [1.], [1.], [1.]] 

    imagefile, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10, a11,a12,a13,a14,a15,a16,a17,a18,a19,a20, a21,a22,a23,a24,a25,a26,a27,a28,a29,a30, a31,a32,a33,a34,a35,a36,a37 = tf.decode_csv(line,record_defaults=record_defaults)

    label_decoded = tf.stack([a1,a2,a3])
    image_decoded = tf.image.decode_jpeg(tf.read_file(imagefile),channels=3)

    image_cast = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])


    # similary tf.placeholder
    # Training batch set
    image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='X')
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='Y')


    ### Graph 
    conv1 = conv_layer(X, 3, 32, name='conv1', stddev=std_dev)
    conv2 = conv_layer(conv1, 32, 64, name='conv2', stddev=std_dev)
    conv3 = conv_layer(conv2, 64, 128, name='conv3', stddev=std_dev )
    conv4 = conv_layer(conv3, 128, 256, name='conv4', stddev=std_dev)
    conv5 = conv_layer(conv4, 256, 512, name='conv5', stddev=std_dev)
    conv6 = conv_layer(conv5, 512, 1024, name='conv6', stddev=std_dev)
    conv7 = conv_layer(conv6, 1024, 2048, name='conv7', stddev=std_dev)
    fc1   = fc_layer(conv7, 2048, 2048, name='fc1', stddev=std_dev)
    fc2   = fc_layer(fc1, 2048, 3, name='fc2', stddev=std_dev)
    logits = fc2

    with tf.name_scope('xent') :
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) 
	cost_hist = tf.summary.scalar('cost', cost) 

    with tf.name_scope('train') :
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
	    cost_value, _, summ, dist1, acc = sess.run([cost, optimizer, summary, dist, accuracy], feed_dict={X: batch_x, Y: batch_y})

	writer.add_summary(summ, epoch)
	print "epoch[%d] : %f " % (epoch, acc) 
	    
    coord.request_stop()
    coord.join(threads) 

def main() :
    for learning_rate in [1.0*1E-4, 4.0*1E-4, 7.0*1E-4]:
	model(learning_rate, 0.01, "param_%f" % (learning_rate))

if __name__ == '__main__':
    main()
