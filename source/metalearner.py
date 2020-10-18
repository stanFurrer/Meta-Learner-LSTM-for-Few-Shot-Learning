import tensorflow as tf
from tensorflow.keras import Model
import source


class MetaLearner(Model):

    def __init__(self, hidden_size, learner, args):
        super(MetaLearner, self).__init__()

        initial_learner_params = learner.get_flat_trainable_params()
        n_learner_params = tf.size(initial_learner_params)

        self.lstm = tf.keras.layers.LSTMCell(units=hidden_size)
        self.lstm.build(tf.TensorShape([n_learner_params, 4]))

        if args.meta_learner == "lstm":
            self.meta_lstm = source.MetaLSTMCell(input_size=hidden_size, n_learner_params=n_learner_params)
            self.meta_lstm.init_cI(initial_learner_params)
        elif args.meta_learner == "full-lstm":
            self.meta_lstm = source.MetaFullLSTMCell(input_size=hidden_size, n_learner_params=n_learner_params)
            self.meta_lstm.init_cI(initial_learner_params)
        elif args.meta_learner == "gru":
            self.meta_lstm = source.MetaGRUCell(input_size=hidden_size, n_learner_params=n_learner_params)
            self.meta_lstm.init_cI(initial_learner_params)
        elif args.meta_learner == "sgd":
            self.meta_lstm = source.MetaSGDCell(n_learner_params=n_learner_params)
            self.meta_lstm.init_cI(initial_learner_params)
        else:
            raise Exception()

    @tf.function
    def call(self, inputs):
        """:param inputs
            [loss, grad_prep, grad, hs]
                loss (Tensor of size [1, 2])
                grad_prep (Tensor of size [n_learner_params, 2])
                grad (Tensor of size [n_learner_params])
                hs = [lstm_cx, [metalstm_fn, metalstm_in, metalstm_cn]]
            :returns learner_params, [lstm_hx, meta_lstm_hs]
                learner_params (Tensor of size [n_learner_params])
                [lstm_cx, meta_lstm_hs]
        """
        loss, grad_prep, grad, hs = inputs
        loss = tf.broadcast_to(loss, tf.shape(grad_prep))
        lstm_input = tf.concat([loss, grad_prep], 1)  # [n_learner_params, 4]

        if hs[0] is None:
            hs = [self.lstm.get_initial_state(inputs=lstm_input), [None, tf.reshape(hs[1], (-1, 1))]]

        lstm_hx, lstm_cx = self.lstm(lstm_input, hs[0])

        learner_params, meta_lstm_hs = self.meta_lstm([lstm_hx, tf.reshape(grad, [-1, 1]), hs[1]])

        return learner_params, [lstm_cx, meta_lstm_hs]
