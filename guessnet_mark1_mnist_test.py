from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class TestGuessNetMark1Mnist(tf.test.TestCase):
    def test___guess(self):
        original = tf.constant([
            [[2, 4], [2, 4], [4, 6], [2, 4]],
            [[4, 6], [2, 4], [3, 5], [4, 6]],
            [[1, 3], [4, 6], [1, 3], [1, 3]]
        ])
        mean_array = [
            [3, 3, 5, 3],
            [5, 3, 4, 5],
            [2, 5, 2, 2]
        ]
        mean_min_indexes_array = [0, 1, 0]
        min_original_array = [[2, 4], [2, 4], [1, 3]]
        with self.test_session():
            # Mean
            mean = tf.reduce_mean(original, 2)
            self.assertAllEqual(mean.eval(), mean_array)
            # Min indexes
            mean_min_indexes = tf.argmin(mean, 1)
            self.assertAllEqual(mean_min_indexes.eval(), mean_min_indexes_array)
            # Slice indexes
            slice = lambda x: (tf.slice(x[0],[x[1],0], [1,2]), 0)
            min_original_stacked = tf.map_fn(slice, (original, mean_min_indexes))[0]
            min_original = tf.squeeze(min_original_stacked, axis=1)

            print(min_original.eval())

        #s = tf.slice(c, [i,0], [1,4])
        #i_2 = tf.constant([[1],[1],[1]])
        #s_2 = tf.gather_nd(c, i_2)
        #print(sess.run([s_2]))
