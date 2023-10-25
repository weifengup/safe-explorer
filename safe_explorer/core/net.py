import tensorflow as tf


class Net(tf.keras.Model):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims,
                 init_bound,
                 initializer,
                 last_activation):
        super(Net, self).__init__()

        self._initializer = initializer
        self._last_activation = last_activation

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        print(_layer_dims)
        self._layers = [tf.keras.layers.Dense(layer_dim, kernel_initializer=initializer) for layer_dim in _layer_dims]

        # self._init_weights(init_bound)

    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        for layer in self._layers[:-1]:
            print(layer)
            # weights = layer.get_weights()
            # print(weights)
            # weights[0] = self._initializer(weights[0])
            # layer.set_weights(weights)

        # Initialize the last layer with a uniform initializer
        weights = self._layers[-1].get_weights()
        weights[0] = tf.random.uniform(shape=weights[0].shape, minval=-bound, maxval=bound)
        self._layers[-1].set_weights(weights)

    def call(self, inp):
        out = inp

        for layer in self._layers[:-1]:
            out = tf.nn.relu(layer(out))

        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)

        return out
