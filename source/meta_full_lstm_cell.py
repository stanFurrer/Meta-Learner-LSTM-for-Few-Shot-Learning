import tensorflow as tf
from tensorflow.keras import Model


class MetaFullLSTMCell(Model):
    def __init__(self, input_size, n_learner_params):
        super(MetaFullLSTMCell, self).__init__()
        self.input_size = input_size
        self.n_learner_params = n_learner_params

        self.WF = tf.Variable(tf.zeros([input_size + 2, 1]), name="meta_lstm/WF")
        self.WI = tf.Variable(tf.zeros([input_size + 2, 1]), name="meta_lstm/WI")
        self.WO = tf.Variable(tf.zeros([input_size + 2, 1]), name="meta_lstm/WO")

        self.cI = tf.Variable(tf.zeros([self.n_learner_params]), name="meta_lstm/cI")

        self.bF = tf.Variable(tf.zeros(1), name="meta_lstm/bF")
        self.bI = tf.Variable(tf.zeros(1), name="meta_lstm/bI")
        self.bO = tf.Variable(tf.zeros(1), name="meta_lstm/bO")

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.trainable_variables:
            weight.assign(tf.random.uniform(tf.shape(weight), minval=-0.01, maxval=0.01))

        # want initial forget value to be high and input value to be low so that
        #  model starts with gradient descent
        self.bF.assign(tf.random.uniform(tf.shape(self.bF), minval=4, maxval=6))
        self.bI.assign(tf.random.uniform(tf.shape(self.bI), minval=-5, maxval=-4))
        self.bO.assign(tf.zeros_like(self.bO))

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
                    h (Tensor of size [n_learner_params, 1]): hidden state
                    c (Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        x_all, grad, hx = inputs

        if hx[0] is None:
            h_prev = tf.zeros([self.n_learner_params, 1])
            hx = [h_prev, hx[-1]]

        h_prev, c_prev = hx

        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, h_{t-1}] + b_f)
        f_next = tf.sigmoid(tf.matmul(tf.concat([x_all, c_prev, h_prev], -1), self.WF) + tf.broadcast_to(self.bF, tf.shape(h_prev)))
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, h_{t-1}] + b_i)
        i_next = tf.sigmoid(tf.matmul(tf.concat([x_all, c_prev, h_prev], -1), self.WI) + tf.broadcast_to(self.bI, tf.shape(h_prev)))
        # o_t = sigmoid(W_0 * [grad_t, loss_t, theta_{t-1}, h_{t-1}] + b_o)
        o_next = tf.sigmoid(tf.matmul(tf.concat([x_all, c_prev, h_prev], -1), self.WO) + tf.broadcast_to(self.bO, tf.shape(h_prev)))
        # next cell/params
        c_next = tf.math.multiply(f_next, c_prev) - tf.math.multiply(i_next, grad)
        h_next = tf.math.multiply(o_next, tf.tanh(c_next))

        return c_next, [h_next, c_next]