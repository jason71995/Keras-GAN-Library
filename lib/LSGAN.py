'''
    Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities
    Ref:
        - https://arxiv.org/abs/1701.06264
'''

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D,GlobalAveragePooling2D,LeakyReLU,UpSampling2D
from keras.optimizers import Adam
from keras.layers import Input

def build_generator(input_shape):
    model = Sequential()

    model.add(Conv2D(512,(3,3),input_shape=input_shape,padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(256,(3,3),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(3,(3,3),padding="same",activation="tanh"))
    return model


def build_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(64,(3,3),input_shape=input_shape,padding="same"))
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

def get_training_function(batch_size,noise_size,image_size,generator,discriminator):

    real_image = Input(image_size)
    noise = K.random_normal((batch_size,) + noise_size,0.0,1.0,"float32")
    fake_image = generator(noise)

    LAMBA = 0.0002

    pred_real = discriminator(real_image)
    pred_fake = discriminator(fake_image)

    d_loss = K.mean(K.maximum(pred_real - pred_fake + LAMBA * K.sum(K.abs(real_image-fake_image),axis=[1,2,3]), 0.0))
    g_loss = K.mean(pred_fake)

    d_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(discriminator.trainable_weights, [], d_loss)
    d_train = K.function([real_image, K.learning_phase()], [d_loss], d_training_updates)

    g_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(generator.trainable_weights, [], g_loss)
    g_train = K.function([K.learning_phase()], [g_loss], g_training_updates)

    return d_train,g_train