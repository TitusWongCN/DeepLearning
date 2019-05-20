# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5

import tensorflow as tf

# 创建一个常量op
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
# 创建一个乘法op
product = tf.matmul(m1, m2)
# 定义一个会话　启动默认图
sess = tf.Session()
# 调用sess的run方法来执行矩阵乘法
# run(product)触发了图中三个op
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
