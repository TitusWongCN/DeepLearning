# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5

import tensorflow as tf

# 创建一个常量op
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: 1.0, input2: 5.0})
    print(result)
