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
lr = tf.Variable(0.001, dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L2_drop, W3 + b3))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for _ in range(201):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** _)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
        
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:1.0})
        print('Iter:', _, '\t train:', str(train_acc), '\ttest:', str(test_acc))

'''
神经元个数： 784, 2000, 2000, 1000, 10
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
'''
神经元个数： 784, 500， 300, 10
Iter: 0          train: 0.9528  test: 0.9471
Iter: 1          train: 0.96976364      test: 0.9634
Iter: 2          train: 0.97803634      test: 0.9677
Iter: 3          train: 0.97983634      test: 0.9685
Iter: 4          train: 0.98267275      test: 0.9691
Iter: 5          train: 0.9863091       test: 0.9713
Iter: 6          train: 0.9874727       test: 0.9732
Iter: 7          train: 0.9884909       test: 0.9736
Iter: 8          train: 0.99047273      test: 0.9752
'''
'''
神经元个数： 784, 500， 300, 10
学习率动态调整 ： lr
Iter: 0          train: 0.9523091       test: 0.9484
Iter: 1          train: 0.9700909       test: 0.9646
Iter: 2          train: 0.97801816      test: 0.968
Iter: 3          train: 0.9831273       test: 0.9699
Iter: 4          train: 0.9838727       test: 0.9714
Iter: 5          train: 0.98843634      test: 0.9742
Iter: 6          train: 0.9896909       test: 0.9753
Iter: 7          train: 0.9913273       test: 0.9753
Iter: 8          train: 0.9908  test: 0.9754
'''