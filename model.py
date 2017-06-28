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
import time
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


prepare_data.prepare_image()

### Hyper parameter
IMAGE_WIDTH =  128
IMAGE_HEIGHT = 128
KEEP_PROB = 0.7
TRAIN_EPOCH = 300 #500
BATCH_SIZE = 50
NUM_TOTAL_TRAINING_DATA = 61578
NUM_TOTAL_VALID_DATA = 100
NUM_THREADS = 4
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 37
FILTER_SIZE = 3
POOLING_SIZE = 2

def conv_layer(input, size_in, size_out, training=True, name="conv"):
  with tf.name_scope(name):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w = tf.get_variable(name+"_w", shape=[FILTER_SIZE, FILTER_SIZE, size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    h1 = conv + b
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=training, scope=name)
    act = tf.nn.relu(h2, 'relu')
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, is_relu=True, training=True, name="fc"):
  with tf.name_scope(name):
    flat_input = tf.reshape(input, [-1, size_in])
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w = tf.get_variable(name+"_w", shape=[size_in, size_out], initializer=w_init)
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    h1  = tf.matmul(flat_input, w) + b
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=training, scope=name)
    if is_relu :
	act = tf.nn.relu(h2, 'relu')
    else :
	act = h2
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

def decision_tree(input, name='dt') :

    out_l1 = tf.nn.softmax(input[:,0:3], name='q1')
   
    out_l2 = tf.nn.softmax(input[:,3:5], name='q2') 
   
    out_l3 = tf.nn.softmax(input[:,5:7], name='q3') 
   
    out_l4 = tf.nn.softmax(input[:,7:9], name='q4')
  
    out_l5 = tf.nn.softmax(input[:,9:13], name='q5') 

    out_l6 = tf.nn.softmax( input[:,13:15], name='q6') 
     
    out_l7 = tf.nn.softmax(input[:,15:18], name='q7') 
   
    out_l8 = tf.nn.softmax(input[:,18:25], name='q8')
   
    out_l9 = tf.nn.softmax(input[:,25:28], name='q9')
  
    out_l10 = tf.nn.softmax(input[:,28:31], name='q10') 
 
    out_l11 = tf.nn.softmax(input[:,31:37], name='q11') 

    res=tf.concat([out_l1,out_l2,out_l3,out_l4,out_l5,out_l6,out_l7,out_l8,out_l9,out_l10,out_l11],1)
    
    return res


def dt_for_test(input, name='dt') :

    row = tf.shape(input)[0]

    out_l1 = input[:,0:3]  
    res1 = tf.nn.softmax( out_l1, name='q1')
   
    out_l2 = tf.nn.softmax(input[:,3:5], name='q2') 
    res2 = tf.multiply(out_l2, tf.reshape(res1[:,1],(row,1)))
   
    out_l3 = tf.nn.softmax(input[:,5:7], name='q3') 
    res3 = tf.multiply(out_l3, tf.reshape(res2[:,1],(row,1)))
   
    out_l4 = tf.nn.softmax(input[:,7:9], name='q4')
    res4 = tf.multiply(out_l4,tf.reshape(res2[:,1], (row,1)))
  
    out_l5 = tf.nn.softmax(input[:,9:13], name='q5') 
    res5 = tf.multiply(out_l5,tf.reshape(res2[:,1], (row,1)))

    out_l6 = input[:,13:15]   
    res6 = tf.nn.softmax(out_l6, name='q6') 
     
    out_l7 = tf.nn.softmax(input[:,15:18], name='q7') 
    res7 = tf.multiply(out_l7,tf.reshape(res1[:,0], (row,1)))
   
    out_l8 = tf.nn.softmax(input[:,18:25], name='q8')
    res8 = tf.multiply(out_l8,tf.reshape(res6[:,0], (row,1)))
   
    out_l9 = tf.nn.softmax(input[:,25:28], name='q9')
    res9 = tf.multiply(out_l9,tf.reshape(res2[:,0], (row,1)))
  
  
    out_l10 = tf.nn.softmax(input[:,28:31], name='q10') 
    res10 = tf.multiply(out_l10,tf.reshape(res4[:,0], (row,1)))
 
    out_l11 = tf.nn.softmax(input[:,31:37], name='q11') 
    res11 = tf.multiply(out_l11,tf.reshape(res4[:,0], (row,1)))

    res=tf.concat([res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11],1)
    
    return res

def prepare_input(path='') :
     # input your path
    csv_file =  tf.train.string_input_producer([path], name='filename_queue', shuffle=True)       
    csv_reader = tf.TextLineReader()
    _,line = csv_reader.read(csv_file)


    record_defaults = [[""]] + [[1.]] * 37 
    rows = tf.decode_csv(line,record_defaults=record_defaults)
    label_decoded = tf.stack([rows[1:NUM_CLASSES+1]])

    image_decoded = tf.image.decode_jpeg(tf.read_file(rows[0]),channels=3)

    image_cast = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    return image, label_decoded

def y_prime(y) :

    epsilon = 1e-4
    row = y.shape[0]
    y_prime = copy.deepcopy(y)

    # Q1
    y_prime[:,0:3] = y[:,0:3] 

    # Q2
    div = y[:,1].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0

    y_prime[:,3:5] = y[:,3:5] / div
    y_prime[check_zero,3:5] = 0.0

    # Q3
    div = y[:,4].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0

    y_prime[:,5:7] = y[:,5:7] / div
    y_prime[check_zero,5:7] = 0.0

    
    # Q4
    div = y[:,4].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,7:9] = y[:,7:9] / div
    y_prime[check_zero,7:9] = 0.0
 
    # Q5
    div = y[:,4].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,9:13] = y[:,9:13] / div
    y_prime[check_zero,9:13] = 0.0
 
    # Q6
    y_prime[:,13:15] = y[:,13:15] 

    # Q7
    div = y[:,0].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,15:18] = y[:,15:18] / div
    y_prime[check_zero,15:18] = 0.0

    # Q8
    div = y[:,13].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,18:25] = y[:,18:25] / div
    y_prime[check_zero,18:25] = 0.0

    # Q9
    div = y[:,3].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,25:28] = y[:,25:28] / div
    y_prime[check_zero,25:28] = 0.0
    
    # Q10
    div = y[:,7].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,28:31] = y[:,28:31] / div
    y_prime[check_zero,28:31] = 0.0

    # Q11
    div = y[:,7].reshape(row,1)
    check_zero = (div < epsilon).reshape(row,).tolist()
    div[check_zero] = 1.0
 
    y_prime[:,31:37] = y[:,31:37] / div
    y_prime[check_zero,31:37] = 0.0

    return y_prime
      
def model(learning_rate, hparam) :

    # open session
    tf.reset_default_graph()
    sess = tf.Session()

    # prepare input
    image, label_decoded = prepare_input(prepare_data.MODIFIED_TRAIN_LABEL_CSV_FILE)

    valid_image, valid_label_decoded = prepare_input(prepare_data.MODIFIED_VALID_LABEL_CSV_FILE)

    # similary tf.placeholder
    # Training batch set
    image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)


    valid_image_batch, valid_label_batch = tf.train.shuffle_batch([valid_image, valid_label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='X')
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='Y')
    Y_prime = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='Y_prime')
    is_training = tf.placeholder(tf.bool, name='is_training')

    ### Graph 
    conv1 = conv_layer(X, 3, 32, training = is_training, name='conv1')
    conv2 = conv_layer(conv1, 32, 64, training = is_training, name='conv2')
    conv3 = conv_layer(conv2, 64, 128, training = is_training, name='conv3')
    conv4 = conv_layer(conv3, 128, 256, training = is_training, name='conv4')
    conv5 = conv_layer(conv4, 256, 512, training = is_training, name='conv5')
    conv6 = conv_layer(conv5, 512, 1024, training = is_training, name='conv6')
    conv7 = conv_layer(conv6, 1024, 2048, training = is_training, name='conv7')
    fc1   = fc_layer(conv7, 2048, 2048, training = is_training, name='fc1')
    fc2   = fc_layer(fc1, 2048, NUM_CLASSES, is_relu=False, training = is_training, name='fc2')


    with tf.name_scope('decision_tree') :
        logits = decision_tree(fc2, name='dt')
	logits_for_test = dt_for_test(fc2, name='dt')
	#logits = tf.nn.softmax(fc2, name=None)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.name_scope('train') :
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_prime)) 
	cost_hist = tf.summary.scalar('cost', cost) 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('distance') :
	dist = tf.reduce_mean(tf.square(tf.subtract(logits_for_test, Y)), axis=1)
	dist_train = tf.reduce_mean(tf.sqrt(dist))
	dist_train_hist = tf.summary.scalar('dist_train', dist_train)
	dist_valid = tf.reduce_mean(tf.sqrt(dist))
	dist_valid_hist = tf.summary.scalar('dist_valid', dist_valid)

    with tf.name_scope('accuracy') :
	correct_prediction = tf.equal(tf.argmax(logits_for_test,1), tf.argmax(Y,1))	
	acc_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	acc_train_hist = tf.summary.scalar('acc_train', acc_train)
	acc_valid = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	acc_valid_hist = tf.summary.scalar('acc_valid', acc_valid)
	
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/'+hparam)
    writer.add_graph(sess.graph)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(TRAIN_EPOCH):
	
	## TRAINING
	total_batch = int(NUM_TOTAL_TRAINING_DATA/BATCH_SIZE)
	for i in range(total_batch):
	    batch_x, batch_y = sess.run([image_batch, label_batch])
	    batch_y = batch_y.reshape(BATCH_SIZE, NUM_CLASSES)
	    batch_y_prime = y_prime(batch_y)
	    opt, acc_train_ = sess.run([optimizer, acc_train], 
		feed_dict={X: batch_x, Y: batch_y, Y_prime : batch_y_prime, is_training: True})
	    cost_h, dist_train_h, acc_train_h = sess.run([cost_hist, dist_train_hist, acc_train_hist], 
		feed_dict={X: batch_x, Y: batch_y, Y_prime : batch_y_prime, is_training: True})

	writer.add_summary(cost_h, epoch)
	writer.add_summary(dist_train_h, epoch)
	writer.add_summary(acc_train_h, epoch)

	## VALIDATION
	total_batch = int(NUM_TOTAL_VALID_DATA/BATCH_SIZE) 
	for i in range(total_batch):
	    batch_x, batch_y = sess.run([valid_image_batch, valid_label_batch])
	    batch_y = batch_y.reshape(BATCH_SIZE, NUM_CLASSES)
	    batch_y_prime = y_prime(batch_y)
	    acc_valid_ = sess.run(acc_valid, 
		feed_dict={X: batch_x, Y: batch_y, Y_prime : batch_y_prime, is_training: True})
	    dist_valid_h, acc_valid_h = sess.run([dist_valid_hist, acc_valid_hist], 
		feed_dict={X: batch_x, Y: batch_y, Y_prime : batch_y_prime, is_training: True})

	writer.add_summary(dist_valid_h, epoch)
	writer.add_summary(acc_valid_h, epoch)


	#writer.add_summary(summ, epoch)
	print "epoch[%d]: acc(%f,%f) dist(%f,%f)" % (epoch, acc_train_, acc_valid_, 0.0, 0.0) 

    ## TEST
    #run_test()   
	    
    coord.request_stop()
    coord.join(threads) 

def main() :
    count = 0
    for learning_rate in [ 1.0*1E-5  ] :
	strtime = time.strftime('%y%m%d_%H%M')
	model(learning_rate, "log_%s_%f_%d" % (strtime, learning_rate, count))
	count += 1

if __name__ == '__main__':
    main()
