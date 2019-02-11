'''
    Spectral Normalization for Generative Adversarial Networks
    Ref:
        - https://arxiv.org/abs/1802.05957
        - https://github.com/pfnet-research/sngan_projection/tree/master/source
'''

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D,LeakyReLU,Conv2DTranspose, Conv2D
from keras.optimizers import Adam

from keras.layers.convolutional import _Conv
from keras.legacy import interfaces
from keras.engine import InputSpec

def build_generator(input_shape):
    model = Sequential()

    model.add(Conv2DTranspose(512,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(3,(3,3),padding="same",activation="tanh"))
    return model


def build_discriminator(input_shape):
    model = Sequential()

    model.add(SNConv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(512,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(1,(3,3),padding="same"))
    model.add(GlobalAveragePooling2D())

    return model

def build_functions(batch_size, noise_size, image_size, generator, discriminator):

    noise = K.random_normal((batch_size,) + noise_size,0.0,1.0,"float32")
    real_image = K.placeholder((batch_size,) + image_size)
    fake_image = generator(noise)

    d_input = K.concatenate([real_image, fake_image], axis=0)
    pred_real, pred_fake = tf.split(discriminator(d_input), num_or_size_splits = 2, axis = 0)

    d_loss = K.mean(K.maximum(0., 1 - pred_real)) + K.mean(K.maximum(0., 1 + pred_fake))
    g_loss = -K.mean(pred_fake)

    d_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(d_loss, discriminator.trainable_weights)
    d_train = K.function([real_image, K.learning_phase()], [d_loss], d_training_updates)

    g_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(g_loss, generator.trainable_weights)
    g_train = K.function([real_image, K.learning_phase()], [g_loss], g_training_updates)

    return d_train,g_train

class SNConv2D(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())
