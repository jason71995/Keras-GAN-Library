'''
    Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities
    Ref:
        - https://arxiv.org/abs/1701.06264
'''

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D,GlobalAveragePooling2D,LeakyReLU,Conv2DTranspose
from keras.optimizers import Adam

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

    model.add(Conv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1,(3,3),padding="same"))
    model.add(GlobalAveragePooling2D())

    return model

def build_functions(batch_size, noise_size, image_size, generator, discriminator):

    noise = K.random_normal((batch_size,) + noise_size,0.0,1.0,"float32")
    real_image = K.placeholder((batch_size,) + image_size)
    fake_image = generator(noise)

    LAMBA = 0.0002

    d_input = K.concatenate([real_image, fake_image], axis=0)
    pred_real, pred_fake = tf.split(discriminator(d_input), num_or_size_splits = 2, axis = 0)

    d_loss = K.mean(K.maximum(pred_real - pred_fake + LAMBA * K.sum(K.abs(real_image-fake_image),axis=[1,2,3]), 0.0))
    g_loss = K.mean(pred_fake)

    d_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(d_loss, discriminator.trainable_weights)
    d_train = K.function([real_image, K.learning_phase()], [d_loss], d_training_updates)

    g_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(g_loss, generator.trainable_weights)
    g_train = K.function([real_image, K.learning_phase()], [g_loss], g_training_updates)

    return d_train,g_train