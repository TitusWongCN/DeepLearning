# -*- coding = utf-8 -*-
# !/home/titus/Work/Python/venv/bin/python3.5
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
batch_size = 100
max_steps = 1001
image_num = 3000
DIR = '/'

sess = tf.Session()
embedding = tf.Variable(tf.stack(mnist.stack(mnist.test.images[:image_num]), trainable= False, name= 'embedding'))

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shape_input, 10)

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='w')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(prediction - y))
    tf.summary.scalar('loss', loss)
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(lables[i]) + '\n')

merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embedding.add()
embed.tensor_name = embedding.name 
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    
    if i%100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter:', i, ', Testing Accuracy= ', acc)
        
    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    sess.close()
