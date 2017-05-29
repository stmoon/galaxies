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
TRAIN_EPOCH =1000
BATCH_SIZE = 50
NUM_TOTAL_TRAINING_DATA = 61578
NUM_THREADS = 4
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 3
FILTER_SIZE = 2
POOLING_SIZE = 2


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

### Graph part
print "original: ", X
filter1 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 3, 32], stddev=0.1))
L1 = tf.nn.conv2d(X, filter1, strides=[1, 1, 1, 1], padding='SAME')
# print L1
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L1=tf.nn.dropout(L1,KEEP_PROB)
print "after 1-layer: ", L1


filter2 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 32, 64], stddev=0.1))
L2 = tf.nn.conv2d(L1, filter2, strides=[1, 1, 1, 1], padding='SAME')
# print L2
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L2=tf.nn.dropout(L2,KEEP_PROB)

print "after 2-layer: ", L2


filter3 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, filter3, strides=[1, 1, 1, 1], padding='SAME')
# print L3
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L3=tf.nn.dropout(L3,KEEP_PROB)

print "after 3-layer: ", L3


filter4 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 128, 256], stddev=0.01))
L4 = tf.nn.conv2d(L3, filter4, strides=[1, 1, 1, 1], padding='SAME')
# print L4
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L4=tf.nn.dropout(L4,KEEP_PROB)

print "after 4-layer: ", L4


filter5 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 256, 512], stddev=0.001))
L5 = tf.nn.conv2d(L4, filter5, strides=[1, 1, 1, 1], padding='SAME')
# print L5
L5 = tf.nn.relu(L5)
L5 = tf.nn.max_pool(L5, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L5=tf.nn.dropout(L5,KEEP_PROB)

print "after 5-layer: ", L5


filter6 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 512, 1024], stddev=0.001))
L6 = tf.nn.conv2d(L5, filter6, strides=[1, 1, 1, 1], padding='SAME')
# print L6
L6 = tf.nn.relu(L6)
L6 = tf.nn.max_pool(L6, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L6=tf.nn.dropout(L6,KEEP_PROB)

print "after 6-layer: ", L6

filter7 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 1024, 2048], stddev=0.01))
L7 = tf.nn.conv2d(L6, filter7, strides=[1, 1, 1, 1], padding='SAME')
# print L7
L7 = tf.nn.relu(L7)
L7 = tf.nn.max_pool(L7, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')

print "after 7-layer: ", L7



print "========================================================================================="

L7 = tf.reshape(L7, [-1, 1*1*2048])
print "reshape for fully: ", L7


flat_W1 = tf.get_variable("flat_W", shape=[1*1*2048, NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
w1_hist = tf.histogram_summary('W1', flat_W1)
b1 = tf.Variable(tf.random_normal([NUM_CLASSES]))
b1_hist = tf.histogram_summary('b1', b1)
logits = tf.matmul(L7, flat_W1) + b1
logits_hist = tf.histogram_summary('logit', logits)


param_list = [filter1, filter2, filter3, filter4, filter5, filter6, filter7, flat_W1, b1]
saver = tf.train.Saver(param_list)


print "========================================================================================="
print "logits: ", logits
print "Y: ", Y
print "========================================================================================="



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
cost_hist = tf.scalar_summary('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('/tmp/logs')

with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
 
    for epoch in range(TRAIN_EPOCH):
        avg_cost = 0
        total_batch = int(NUM_TOTAL_TRAINING_DATA/BATCH_SIZE)

        ## TRAINING
        for i in range(total_batch):
            batch_x, batch_y = sess.run([image_batch, label_batch])
            
            batch_y = batch_y.reshape(BATCH_SIZE, NUM_CLASSES)
            
            summ, cost_value, _ = sess.run([merged, cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += cost_value / total_batch

	writer.add_summary(summ,epoch)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        saver.save(sess, 'log/model'+str(epoch)+'.ckpt', global_step=100)

	## TEST ACCURACY
	test_batch = []
	test_y = []
	test_path = prepare_data.RESIZE_TRAIN_DATA_PATH
	test_y_path = prepare_data.MODIFIED_TRAIN_LABEL_CSV_PATH

	reader = csv.reader(open(test_y_path), delimiter=',')
         
	for row in reader :
	    file_path = row[0]
	    img = cv2.imread(file_path)
	    test_batch.append(img)
	    test_y.append([float(row[1]), float(row[2]), float(row[3])])

	    if  len(test_y) == 50 : 
		break;

	input_batch = np.array(test_batch)
	input_batch = input_batch.reshape(50, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
	y_batch = np.array(test_y)
	y_batch = y_batch.reshape(50,3)

	pred = sess.run(logits, feed_dict={X: input_batch})

	pred1 = pred
	pred1[pred1<0] = 0.0
	s = np.expand_dims(np.sum(pred1,1)+1e-12,1)
	pred1 = pred1 / s

	accuracy =  np.average(np.sqrt(np.sum((pred1-y_batch)**2, axis=1)))
	print('Accuracy : ' ,  accuracy)
    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 50 images test
    test_batch = []
    file_list = []
    test_path = prepare_data.RESIZE_TRAIN_DATA_PATH
    for test_file in os.listdir(test_path)[:50]:
        file_list.append(test_file)
        img = cv2.imread(test_path+'/'+test_file, cv2.IMREAD_GRAYSCALE)
        test_batch.append(img)

    input_batch = np.array(test_batch)
    input_batch = input_batch.reshape(50, IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    pred = sess.run(tf.argmax(logits, 1), feed_dict={X: input_batch})
    print 'predict : ', pred
    print file_list


    # show prediction using pyplot
    fig = plt.figure()
    for i in range(50):
        y = fig.add_subplot(5, 10, i+1)
        show_img = input_batch[i, :, :, 0]
        y.imshow(show_img, cmap='gray')
	y.set_xticks([])
	y.set_yticks([])
        plt.title(pred[i])
    plt.show()


    coord.request_stop()
    coord.join(threads) 

