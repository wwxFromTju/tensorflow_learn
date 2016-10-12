#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf


two = tf.constant(2)
three = tf.constant(3)
five = tf.constant(5)
add = tf.add(two, five)
mul = tf.mul(add, three)

with tf.Session() as sess:
    # with tf.device('/cpu:0'):
        answer = sess.run([add, mul])
        print(answer)