# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for _ in range(201):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
        
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:1.0})
        print('Iter:', _, '\t train:', str(train_acc), '\ttest:', str(test_acc))

'''
Iter: 0          train: 0.73701817      test: 0.738
Iter: 1          train: 0.9534182       test: 0.9496
Iter: 2          train: 0.9597273       test: 0.9547
Iter: 3          train: 0.9646909       test: 0.9581
Iter: 4          train: 0.9693818       test: 0.9617
Iter: 5          train: 0.9678909       test: 0.9596
Iter: 6          train: 0.96761817      test: 0.9599
Iter: 7          train: 0.97163635      test: 0.9629
Iter: 8          train: 0.96696365      test: 0.9603
'''