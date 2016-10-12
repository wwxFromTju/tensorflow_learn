#!/usr/bin/env
# encoding=utf-8

import tensorflow as tf


hold_1 = tf.placeholder(tf.float32)
hold_2 = tf.placeholder(tf.float32)
mul = tf.mul(hold_1, hold_2)

with tf.Session() as sess:
    print(sess.run([mul], feed_dict={hold_1: [4], hold_2: [5]}))
