from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from guessnet import GuessNet

import random
import math
import array
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

ONE_HOT_RANDOM_SIZE = 30
GUESS_COUNT_PER_INPUT = 20
BATCH_SIZE = 1


class GuessNetMark1Mnist:
    def __init__(self):
        self.original = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        self.one_hot_random = tf.placeholder(tf.float32, [None, GUESS_COUNT_PER_INPUT, ONE_HOT_RANDOM_SIZE])

    def layers(self):
        compressed = tf.reshape(self.one_hot_random, shape=[BATCH_SIZE * GUESS_COUNT_PER_INPUT, ONE_HOT_RANDOM_SIZE])
        guesses_compressed = self.__guess(compressed)
        tf.summary.image('guesses', tf.reshape(guesses_compressed, [BATCH_SIZE*GUESS_COUNT_PER_INPUT, IMAGE_SIZE, IMAGE_SIZE, 1]), max_outputs=40)
        guesses = tf.reshape(guesses_compressed, shape=[BATCH_SIZE, GUESS_COUNT_PER_INPUT, IMAGE_PIXELS])
        return self.__generate(guesses, self.__guess_mask(self.__error(guesses)))

    def __guess(self, one_hot_random_compressed):
        W1 = tf.Variable(tf.truncated_normal([ONE_HOT_RANDOM_SIZE, 1024], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        r1 = tf.nn.relu(tf.matmul(one_hot_random_compressed, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal([1024, IMAGE_PIXELS], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[IMAGE_PIXELS]))
        r2 = tf.nn.sigmoid(tf.matmul(r1, W2) + b2)
        return r2

    def __error(self, guesses):
        original_x_guess_count = tf.tile(tf.expand_dims(self.original, 0), [1, GUESS_COUNT_PER_INPUT, 1])
        return tf.reduce_mean(tf.square(guesses - original_x_guess_count), axis=2)

    def __guess_mask(self, error):
        maximum_error = tf.expand_dims(tf.reduce_max(error, axis=1), 1)
        grounded_error = maximum_error - error
        scale = tf.expand_dims(tf.reduce_sum(grounded_error, axis=1), 1)
        return grounded_error / scale

    def __generate(self, guesses, guesses_mask):
        masked = tf.matmul(tf.expand_dims(guesses_mask, 1),guesses)
        return tf.squeeze(masked, axis=1)


net = GuessNetMark1Mnist()
layers = net.layers()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=net.original, logits=layers))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.image('result', tf.reshape(layers, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]))
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('/code/logs', graph=tf.get_default_graph())

    # Train
    for i in range(100000):
        batch_original, _ = mnist.train.next_batch(BATCH_SIZE)
        feed_dict = {
            net.one_hot_random: GuessNet.get_one_hot_random(shape=[BATCH_SIZE, GUESS_COUNT_PER_INPUT, ONE_HOT_RANDOM_SIZE]),
            net.original: batch_original
        }
        #print(sess.run([layers], feed_dict=feed_dict))
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        error = sess.run(cross_entropy, feed_dict=feed_dict)
        writer.add_summary(summary, i)
        writer.flush()
        print(error)
        time.sleep(1*60)