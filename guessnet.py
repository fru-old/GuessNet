import tensorflow as tf

import random
import math


class GuessNet:
    @staticmethod
    def __get_row_one_hot_random(size):
        fraction, integer = math.modf(random.random() * size)
        result = [0.0 for col in range(size)]
        result[int(integer)] = 1 - fraction
        result[(int(integer) + 1) % size] = fraction
        return result

    @staticmethod
    def get_one_hot_random(shape=[]):
        if len(shape) == 1:
            return GuessNet.__get_row_one_hot_random(shape[0])
        else:
            return [GuessNet.get_one_hot_random(shape[1:]) for c in range(shape[0])]
