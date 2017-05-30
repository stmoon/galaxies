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
TRAIN_EPOCH = 1000
BATCH_SIZE = 10 #50
NUM_TOTAL_TRAINING_DATA = 100 #61578
NUM_THREADS = 4
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 3
FILTER_SIZE = 2
POOLING_SIZE = 2


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.histogram_summary(name+"_weights", w)
    tf.histogram_summary(name+"_biases", b)
    tf.histogram_summary(name+"_activations", act)
    return tf.nn.max_pool(act, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    flat_input = tf.reshape(input, [-1, size_in])
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(flat_input, w) + b)
    tf.histogram_summary(name+"_weights", w)
    tf.histogram_summary(name+"_biases", b)
    tf.histogram_summary(name+"_activations", act)
    return act


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

label_decoded = tf.pack([a1,a2,a3])
image_decoded = tf.image.decode_jpeg(tf.read_file(imagefile),channels=3)

image_cast = tf.cast(image_decoded, tf.float32)
image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])


# similary tf.placeholder
# Training batch set
image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='X')
Y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='Y')

### open session
sess = tf.Session()

### Graph 
conv1 = conv_layer(X, 3, 32, name='conv1')
conv2 = conv_layer(conv1, 32, 64, name='conv2')
conv3 = conv_layer(conv2, 64, 128, name='conv3')
conv4 = conv_layer(conv3, 128, 256, name='conv4')
conv5 = conv_layer(conv4, 256, 512, name='conv5')
conv6 = conv_layer(conv5, 512, 1024, name='conv6')
conv7 = conv_layer(conv6, 1024, 2048, name='conv7')
fc1   = fc_layer(conv7, 2048, 3, name='fc1')
logits = fc1


with tf.name_scope('xent') :
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    cost_hist = tf.scalar_summary('cost', cost)

with tf.name_scope('train') :
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

with tf.name_scope('accuracy') :
    accuracy = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(logits, Y)), reduction_indices=1))
    accuracy = tf.reduce_mean(accuracy)
    acc_hist = tf.scalar_summary('accuracy', accuracy)

summary  = tf.merge_all_summaries()
sess.run(tf.initialize_all_variables())
writer = tf.train.SummaryWriter('/tmp/logs/2')
writer.add_graph(sess.graph)


## RUN
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for epoch in range(TRAIN_EPOCH):
    total_batch = int(NUM_TOTAL_TRAINING_DATA/BATCH_SIZE)
    for i in range(total_batch):
        batch_x, batch_y = sess.run([image_batch, label_batch])
        batch_y = batch_y.reshape(BATCH_SIZE, NUM_CLASSES)
        cost_value, _, summ, acc = sess.run([cost, optimizer, summary, accuracy], feed_dict={X: batch_x, Y: batch_y})

    writer.add_summary(summ, epoch)
    print "epoch : %d" % (epoch) 
        
coord.request_stop()
coord.join(threads) 


'''
if __name__ == '__main__':
  main()
'''
