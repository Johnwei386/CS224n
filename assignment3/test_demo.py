# _*_ coding:utf8 _*_

import tensorflow as tf
import numpy as np

def test_boolean_mask():
    with tf.Graph().as_default():
        preds = tf.Variable(np.random.randint(30, size=(2, 3, 5)))
        label = tf.Variable(np.random.randint(30, size=(2, 3)))
        mask = tf.constant([True, True, False, True, True, False], shape=[2, 3])
        preds_mask = tf.boolean_mask(preds, mask)
        label_mask = tf.boolean_mask(label, mask)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(preds))
            print(sess.run(mask))
            print(sess.run(label))
            print(sess.run(preds_mask))
            print(sess.run(label_mask))

test_boolean_mask()
