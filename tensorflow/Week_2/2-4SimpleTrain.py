# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data * 2 + 5

k = tf.Variable(0.0)
b = tf.Variable(0.0)
y = k * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for times in range(201):
        sess.run(train)
        if times % 20 == 0:
            print(times, sess.run([k, b]))
