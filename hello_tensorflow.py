#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf

hello_op = tf.constant('Hello Tensorflow!')

a = tf.constant(1)
b = tf.constant(2)
compute_op = tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(hello_op))
    print(sess.run(compute_op))