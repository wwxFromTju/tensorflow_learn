#!/usr/bin/env python
# encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = 'iris_test.csv'

training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

new_sample = np.array([[6.4, 3.2, 4.5, 1.5],
                       [5.8, 3.1, 5.0, 1.7]], dtype=float)

y = classifier.predict(new_sample)
print('Predictions: {}'.format(str(y)))