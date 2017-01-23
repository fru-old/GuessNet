from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from guessnet import GuessNet

import random
import math
import array

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

ONE_HOT_RANDOM_SIZE = 3
GUESS_COUNT_PER_INPUT = 5
BATCH_SIZE = 1


class GuessNetMark1Mnist:
    def __init__(self):
        self.original = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        self.one_hot_random = tf.placeholder(tf.float32, [None, GUESS_COUNT_PER_INPUT, ONE_HOT_RANDOM_SIZE])

    def layers(self):
        compressed = tf.reshape(self.one_hot_random, shape=[BATCH_SIZE * GUESS_COUNT_PER_INPUT, ONE_HOT_RANDOM_SIZE])
        guesses_compressed = self.__guess(compressed)
        guesses = tf.reshape(guesses_compressed, shape=[BATCH_SIZE, GUESS_COUNT_PER_INPUT, IMAGE_PIXELS])
        return self.__generate(guesses, self.__guess_mask(self.__error(guesses)))

    def __guess(self, one_hot_random_compressed):
        W1 = tf.Variable(tf.truncated_normal([ONE_HOT_RANDOM_SIZE, IMAGE_PIXELS], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[IMAGE_PIXELS]))
        return tf.nn.relu(tf.matmul(one_hot_random_compressed, W1) + b1)

    def __error(self, guesses):
        original_x_guess_count = tf.tile(tf.expand_dims(self.original, 0), [1, GUESS_COUNT_PER_INPUT, 1])
        return tf.reduce_mean(tf.square(guesses - original_x_guess_count), axis=2)

    def __guess_mask(self, error):
        maximum_error = tf.expand_dims(tf.reduce_max(error, axis=1), 1)
        #tf.matmul(maximum_error, error)
        grounded_error = tf.exp(tf.exp(maximum_error - error) - 1) - 1
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
tf.summary.image('guess', tf.reshape(layers, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]))
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('/code/logs', graph=tf.get_default_graph())

    # Train
    for i in range(10000):
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


"""
def layer(batch_size):
    guess = []
    error = []
    for i in range(GUESS_COUNT_PER_INPUT):
        guess.append(cell(tf_get_element_at(one_hot_random, i, [batch_size, ONE_HOT_RANDOM_SIZE])))
        error.append(tf.reduce_mean(500 - tf.abs(guess[i] - original), 1))

    #transposed_error = tf.transpose(tf.pack(error))
    #normalized_error = tf.reduce_max(transposed_error, axis=1) - transposed_error# - tf.reduce_min(transposed_error, axis=1)
    #normalized_error = 1 - tf.truediv((transposed_error - tf.reduce_min(transposed_error, axis=1)), tf.reduce_max(transposed_error, axis=1))

    normalized_error = tf.transpose(tf.pack(error)) #tf.nn.softmax(pack, dim=0)
    guess_reshaped = tf.transpose(guess, perm=[1,2,0])
    error_reshaped = tf.expand_dims(normalized_error, 2)

    return tf.squeeze(tf.matmul(guess_reshaped, error_reshaped), axis=2)



    #return tf.reduce_sum(guess_sum, axis=0)
    #softmax_error = tf.nn.softmax(tf.pack(error), dim=0)
    #return tf.squeeze(tf.matmul(tf.expand_dims(softmax_error, 1), guess), axis=1)

    #error_reshaped = tf.expand_dims(tf.transpose(softmax_error),1)
    #guess_reshaped = tf.transpose(guess, perm=[1,0,2])

    #return tf.squeeze(tf.matmul(error_reshaped, guess_reshaped), axis=1)
    #result before softmax [batch, GUESS_COUNT_PER_INPUT]


layer1 = layer(BATCH_SIZE)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=original, logits=layer1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



tf.summary.scalar('cross_entropy', cross_entropy)
img = tf.reshape(layer1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
tf.summary.image('guess', img)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('/code/logs', graph=tf.get_default_graph())

    # Train
    for i in range(340):
        batch_original, _ = mnist.train.next_batch(BATCH_SIZE)
        feed_dict = {one_hot_random: get_one_hot_random_batch(BATCH_SIZE), original: batch_original}
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        error = sess.run(cross_entropy, feed_dict=feed_dict)
        writer.add_summary(summary, i)
        writer.flush()
        print(error)

    writer.close()
"""