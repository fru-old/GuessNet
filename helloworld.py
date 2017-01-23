from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

ONE_HOT_RANDOM_SIZE = 32
GUESS_COUNT_PER_INPUT = 32

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

original = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
one_hot_random = tf.placeholder(tf.float32, [GUESS_COUNT_PER_INPUT, None, ONE_HOT_RANDOM_SIZE])

def get_row_one_hot_random(size):
    r = math.modf(random.random() * size)
    index = int(r[1])
    result = [0.0 for col in range(size)]
    result[index] = 1 - r[0]
    result[index+1 if index+1 < size else 0] = r[0]
    return result

def get_one_hot_random_batch(batch_size):
    one_hot_random = lambda _: get_row_one_hot_random(ONE_HOT_RANDOM_SIZE)
    return map(lambda _: map(one_hot_random, range(batch_size)), range(GUESS_COUNT_PER_INPUT))

def tf_get_element_at(matrix, i, other):
    return tf.reshape(tf.slice(matrix, [i, 0, 0], [1] + other), other)

def cell(one_hot_random_of_one_guess):
    W1 = tf.Variable(tf.truncated_normal([ONE_HOT_RANDOM_SIZE, 1024], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    r1 = tf.nn.relu(tf.matmul(one_hot_random_of_one_guess, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1024]))
    r2 = tf.nn.relu(tf.matmul(r1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([1024, IMAGE_PIXELS], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[IMAGE_PIXELS]))
    return tf.nn.relu(tf.matmul(r2, W3) + b3)

def cell2():
    one_hot_random_shaped = tf.reshaped(one_hot_random, []

    W1 = tf.Variable(tf.truncated_normal([ONE_HOT_RANDOM_SIZE, 1024], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    r1 = tf.nn.relu(tf.matmul(one_hot_random_of_one_guess, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1024]))
    r2 = tf.nn.relu(tf.matmul(r1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([1024, IMAGE_PIXELS], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[IMAGE_PIXELS]))
    return tf.nn.relu(tf.matmul(r2, W3) + b3)

def layer(batch_size):
    guess = []
    error = []
    for i in range(GUESS_COUNT_PER_INPUT):
        guess.append(cell(tf_get_element_at(one_hot_random, i, [batch_size, ONE_HOT_RANDOM_SIZE])))
        error.append(tf.reduce_mean(500 - tf.abs(guess[i] - original), 1))

    #transposed_error = tf.transpose(tf.pack(error))
    #normalized_error = tf.reduce_max(transposed_error, axis=1) - transposed_error# - tf.reduce_min(transposed_error, axis=1)
    #normalized_error = 1 - tf.truediv((transposed_error - tf.reduce_min(transposed_error, axis=1)), tf.reduce_max(transposed_error, axis=1))
    """
    normalized_error = tf.transpose(tf.pack(error)) #tf.nn.softmax(pack, dim=0)
    guess_reshaped = tf.transpose(guess, perm=[1,2,0])
    error_reshaped = tf.expand_dims(normalized_error, 2)

    return tf.squeeze(tf.matmul(guess_reshaped, error_reshaped), axis=2)
    """


    #return tf.reduce_sum(guess_sum, axis=0)
    #softmax_error = tf.nn.softmax(tf.pack(error), dim=0)
    #return tf.squeeze(tf.matmul(tf.expand_dims(softmax_error, 1), guess), axis=1)

    #error_reshaped = tf.expand_dims(tf.transpose(softmax_error),1)
    #guess_reshaped = tf.transpose(guess, perm=[1,0,2])

    #return tf.squeeze(tf.matmul(error_reshaped, guess_reshaped), axis=1)
    #result before softmax [batch, GUESS_COUNT_PER_INPUT]

BATCH_SIZE = 1
layer1 = layer(BATCH_SIZE)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=original, logits=layer1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

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