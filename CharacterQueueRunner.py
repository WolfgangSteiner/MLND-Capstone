import tensorflow as tf

class CharacterQueueRunner(object):
    def __init__(self):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1])
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.queue = tf.FIFOQueue(capacity=)


    def batch_size(self):
        return 128
