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
GUESS_COUNT_PER_INPUT = 30
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
        #maximum_guess_index = tf.argmax(tf.reduce_mean(guesses, 2), 1)
        return self.__select_minimum(guesses)
        return self.__generate(guesses, self.__guess_mask(self.__error(guesses)))

    def __eluLayer(self, inCount, outCount, inTensor):
        W1 = tf.Variable(tf.truncated_normal([inCount, outCount], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[outCount]))
        return tf.nn.elu(tf.matmul(inTensor, W1) + b1)

    def __guess(self, one_hot_random_compressed):
        W1 = tf.Variable(tf.truncated_normal([ONE_HOT_RANDOM_SIZE, 10], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[10]))
        r1 = tf.nn.relu(tf.matmul(one_hot_random_compressed, W1))  # + b1

        #r1 = self.__eluLayer(ONE_HOT_RANDOM_SIZE, 10, one_hot_random_compressed)
        r2 = self.__eluLayer(10, 50, r1)
        #r3 = self.__eluLayer(ONE_HOT_RANDOM_SIZE, 10, r2)
        #r4 = self.__eluLayer(ONE_HOT_RANDOM_SIZE, 10, r3)
        #r5 = self.__eluLayer(ONE_HOT_RANDOM_SIZE, 10, r4)
        #r6 = self.__eluLayer(ONE_HOT_RANDOM_SIZE, 10, r5)


        W2 = tf.Variable(tf.truncated_normal([50, IMAGE_PIXELS], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[IMAGE_PIXELS]))
        return tf.nn.sigmoid(tf.matmul(r2, W2) + b2)

    def __select_minimum(self, guesses):
        # Mean
        mean = self.__error(guesses)#tf.reduce_mean(guesses, 2)
        # Min indexes
        mean_min_indexes = tf.argmin(mean, 1)
        # Slice indexes
        tf.summary.scalar('mean_min_indexes', tf.squeeze(mean_min_indexes,axis=0))

        def slice_per_batch(x):
            test = tf.cast(x[1], tf.int32)
            return tf.slice(x[0], [test, 0], [1, IMAGE_PIXELS]), 0.0
        min_original_stacked = tf.map_fn(slice_per_batch, (guesses, mean_min_indexes))[0]
        return tf.squeeze(min_original_stacked, axis=1)

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

#train_step = tf.contrib.layers.optimize_loss(cross_entropy,tf.constant(1),0.002,"Adam")
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

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
        time.sleep(1*1)