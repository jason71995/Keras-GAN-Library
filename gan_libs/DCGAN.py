'''
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    Ref:
        - https://arxiv.org/abs/1511.06434
'''

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D,GlobalAveragePooling2D,LeakyReLU,Conv2DTranspose,Activation,Input,BatchNormalization
from keras.optimizers import Adam
from keras.layers import Input

def build_generator(input_shape):
    model = Sequential()

    model.add(Conv2DTranspose(512,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(3,(3,3),padding="same",activation="tanh"))
    return model


def build_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(512,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(1,(3,3),padding="same"))
    model.add(GlobalAveragePooling2D())
    model.add(Activation("sigmoid"))

    return model

def get_training_function(batch_size,noise_size,image_size,generator,discriminator):

    real_image = Input(image_size)

    noise = K.random_normal((batch_size,) + noise_size,0.0,1.0,"float32")
    fake_image = generator(noise)

    pred_real = K.clip(discriminator(real_image),K.epsilon(),1-K.epsilon())
    pred_fake = K.clip(discriminator(fake_image),K.epsilon(),1-K.epsilon())

    d_loss = -(K.mean(K.log(pred_real)) + K.mean(K.log(1-pred_fake)))
    g_loss = -K.mean(K.log(pred_fake))


    # get updates of mean and variance in batch normalization layers
    d_updates = discriminator.get_updates_for([K.concatenate([real_image,fake_image],axis=0)])
    g_updates = generator.get_updates_for([noise])

    d_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(d_loss, discriminator.trainable_weights)
    d_train = K.function([real_image, K.learning_phase()], [d_loss],d_updates + d_training_updates)

    g_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(g_loss, generator.trainable_weights)
    g_train = K.function([K.learning_phase()], [g_loss], g_updates + g_training_updates)

    return d_train,g_train