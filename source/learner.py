import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Dense, Flatten

import source


class Learner(Model):

    def __init__(self, bn_eps, bn_momentum, n_classes, lr, image_size, **kwargs):
        super(Learner, self).__init__(**kwargs)

        input_shape = (image_size, image_size, 3)

        self.input_layer = tf.keras.Input(input_shape)
        self.nn = [Conv2D(32, input_shape=input_shape, kernel_size=3, padding="same", data_format="channels_last"),
                   BatchNormalization(epsilon=bn_eps, momentum=bn_momentum),
                   ReLU(),
                   MaxPool2D(2),

                   Conv2D(32, kernel_size=3, padding="same", data_format="channels_last"),
                   BatchNormalization(epsilon=bn_eps, momentum=bn_momentum),
                   ReLU(),
                   MaxPool2D(2),

                   Conv2D(32, kernel_size=3, padding="same", data_format="channels_last"),
                   BatchNormalization(epsilon=bn_eps, momentum=bn_momentum),
                   ReLU(),
                   MaxPool2D(2),

                   Conv2D(32, kernel_size=3, padding="same", data_format="channels_last"),
                   BatchNormalization(epsilon=bn_eps, momentum=bn_momentum),
                   ReLU(),
                   MaxPool2D(2),

                   Flatten(),
                   Dense(n_classes)]

        # Set up loss, optimizer
        self.loss_object = tf.nn.softmax_cross_entropy_with_logits
        self.train_metrics = [tf.keras.metrics.categorical_accuracy]

        self.out = self.eager_call(self.input_layer)
        super(Learner, self).__init__(inputs=self.input_layer, outputs=self.out, **kwargs)
        self._is_graph_network = True

    def eager_call(self, inputs):
        """calls the model without using TF graph (only for initialization) """
        x = inputs
        for layer in self.nn:
            x = layer(x)
        return x

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.nn:
            x = layer(x, training=training)
        return x

    @tf.function
    def build(self, input_shape):
        """initializes the model as a graph network"""
        self._init_graph_network(inputs=self.input_layer, outputs=self.out)

    @tf.function
    def call_with_grad(self, train_input, train_target):
        """takes a step using the model optimizer"""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            predictions = self.call(train_input, training=True)
            loss = tf.reduce_mean(self.loss_object(train_target, predictions))
            gradients = tf.gradients(loss, self.trainable_variables)
        return loss, gradients

    @tf.function
    def get_flat_trainable_params(self):
        layer_params = []
        for layer in self.trainable_variables:
            layer_params.append(tf.reshape(layer, shape=[-1]))
        return tf.concat(layer_params, axis=0)

    @tf.function
    def set_flat_trainable_params(self, flat_params):
        idx = 0
        for layer in self.trainable_variables:
            layer_size = tf.size(layer)
            layer_shape = tf.shape(layer)
            flat_layer = flat_params[idx:(idx + layer_size)]
            layer.assign(tf.reshape(flat_layer, layer_shape))
            idx = idx + layer_size

    @tf.function
    def set_flat_params_and_predict(self, flat_params, state, val_input):
        idx = 0
        for layer in self.layers:
            attributes = ["kernel", "bias", "gamma", "beta"]
            has_attr = [hasattr(layer, a) for a in attributes]
            for i, attr_name in enumerate(attributes):
                if has_attr[i]:
                    attr = getattr(layer, attr_name)
                    attr_size = tf.size(attr)
                    attr_shape = tf.shape(attr)
                    flat_layer = flat_params[idx:(idx + attr_size)]
                    new_attr = tf.reshape(flat_layer, attr_shape)
                    setattr(layer, attr_name, new_attr)
                    idx += attr_size
        self.set_batch_stats(state)
        return self.call(val_input, training=True)

    @tf.function
    def set_batch_stats(self, params):
        for self_layer, base_layer in zip(self.non_trainable_variables, params):
            self_layer.assign(base_layer)

    @tf.function
    def get_batch_stats(self):
        collection = []
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                collection.append(layer.moving_mean)
                collection.append(layer.moving_variance)
        return collection

    @tf.function
    def reset_batch_stats(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
                layer.moving_variance.assign(tf.ones_like(layer.moving_variance))