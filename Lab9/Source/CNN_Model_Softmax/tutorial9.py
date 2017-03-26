import matplotlib.pyplot as plt
import numpy as np

from not_mnist import NotMNIST
from tensorflow.contrib.session_bundle import exporter



import tensorflow as tf
sess = tf.InteractiveSession()

notmnist_data = NotMNIST()

tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


tf.add_to_collection('variables',W)
tf.add_to_collection('variables',b)

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.global_variables_initializer())


for i in range(1000):
  batch = notmnist_data.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy.eval(feed_dict={x: notmnist_data.test.images, y_: notmnist_data.test.labels})


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.histogram('histogram_Convolution_Weights_1', W_conv1)
tf.summary.histogram('histogram_Convolution_1', h_conv1)

tf.summary.histogram('histogram_Convolution_Weights_2', W_conv2)
tf.summary.histogram('histogram_Convolution_2', h_conv2)

tf.summary.histogram('Keep_prob', keep_prob)
tf.summary.histogram('h_fc1_drop', h_fc1_drop)


tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)

tf.summary.histogram('Accuracy', accuracy)
merged = tf.summary.merge_all()

sess.run(tf.initialize_all_variables())
trainwriter = tf.summary.FileWriter('Output/logs', sess.graph)


for i in range(1000):

    batch1 = notmnist_data.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch1[0], y_: batch1[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch1[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch1[0], y_: batch1[1], keep_prob: 0.5})

print("The Final test accuracy is %g"%accuracy.eval(feed_dict={
    x: notmnist_data.test.images, y_: notmnist_data.test.labels, keep_prob: 1.0}))



export_path = 'Output/model'
print('Exporting model to', export_path)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y_})})

model_exporter.export(export_path, tf.constant(1), sess)