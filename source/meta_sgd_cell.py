import tensorflow as tf
from tensorflow.keras import Model


class MetaSGDCell(Model):
    def __init__(self, n_learner_params):
        super(MetaSGDCell, self).__init__()
        self.n_learner_params = n_learner_params

        self.cI = tf.Variable(tf.zeros([self.n_learner_params]), name="meta_lstm/cI")
        self.lr = tf.Variable(0.01 * tf.ones(1), name="meta_lstm/lr")

    def init_cI(self, cI):
        self.cI.assign(cI)

    @tf.function
    def call(self, inputs):
        """Args:
            :param inputs
            [x_all, grad, hx]:
                x_all (Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (Tensor of size [n_learner_params, 1]): gradients from learner
                hx = [None, c_prev]:
                    None
                    c (Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        _, grad, hx = inputs

        _, c_prev = hx

        # next params
        c_next = c_prev - tf.math.multiply(self.lr, grad)

        return c_next, [None, c_next]