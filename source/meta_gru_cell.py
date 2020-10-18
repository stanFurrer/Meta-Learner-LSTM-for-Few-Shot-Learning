import tensorflow as tf
from tensorflow.keras import Model


class MetaGRUCell(Model):
    def __init__(self, input_size, n_learner_params):
        super(MetaGRUCell, self).__init__()
        self.input_size = input_size
        self.n_learner_params = n_learner_params

        self.WZ = tf.Variable(tf.zeros([input_size + 2, 1]), name="meta_lstm/WZ")
        self.cI = tf.Variable(tf.zeros([self.n_learner_params]), name="meta_lstm/cI")
        self.bZ = tf.Variable(tf.zeros(1), name="meta_lstm/bZ")

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.trainable_variables:
            weight.assign(tf.random.uniform(tf.shape(weight), minval=-0.01, maxval=0.01))

        self.bZ.assign(tf.random.uniform(tf.shape(self.bZ), minval=5, maxval=6))

    def init_cI(self, cI):
        self.cI.assign(cI)

    @tf.function
    def call(self, inputs):
        """Args:
            :param inputs
            [x_all, grad, hx]:
                x_all (Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (Tensor of size [n_learner_params, 1]): gradients from learner
                hx = [f_prev, i_prev, c_prev]:
                    c (Tensor of size [n_learner_params, 1]): flattened learner parameters
        """

        x_all, grad, hx = inputs

        if hx[0] is None:
            z_prev = tf.zeros([self.n_learner_params, 1])
            hx = [z_prev, hx[-1]]

        z_prev, c_prev = hx

        z_next = tf.sigmoid(tf.matmul(tf.concat([x_all, c_prev, z_prev], -1), self.WZ) + tf.broadcast_to(self.bZ, tf.shape(z_prev)))
        c_next = tf.multiply(z_next, c_prev) - tf.multiply(1-z_next, grad)

        return c_next, [z_next, c_next]
