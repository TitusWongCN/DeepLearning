# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5

import tensorflow as tf

# 创建一个常量op
x = tf.Variable([1, 2])
a = tf.constant([3, 3])
# 增加一个减法op和一个加法op
sub = tf.subtract(x, a)
add = tf.add(x, sub)
# 初始化所有变量
init1 = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init1)
    print(sess.run(sub))
    print(sess.run(add))

state = tf.Variable(0, name='counter')
new_value = tf.add(state, 1)
# 赋值op
update = tf.assign(state, new_value)
init2 = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init2)
    print(sess.run(state))
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))
