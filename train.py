import keras
from keras.datasets import cifar10

from gan_libs.DCGAN import build_generator, build_discriminator, build_functions
# from gan_libs.LSGAN import build_generator, build_discriminator, build_functions
# from gan_libs.SNGAN import build_generator, build_discriminator, build_functions
# from gan_libs.WGAN_GP import build_generator, build_discriminator, build_functions

from utils.common import set_gpu_config, predict_images
import numpy as np

set_gpu_config("0",0.5)

epoch = 50
steps = 1000
image_size = (32,32,3)
noise_size = (2,2,32)
batch_size = 16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_data = x_train.shape[0]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train/255)*2-1
x_test = (x_test/255)*2-1
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

generator = build_generator(noise_size)
discriminator = build_discriminator(image_size)
d_train, g_train = build_functions(batch_size, noise_size, image_size, generator, discriminator)

for e in range(epoch):
    for s in range(steps):
        real_images = x_train[np.random.permutation(num_of_data)[:batch_size]]
        d_loss, = d_train([real_images, 1])
        g_loss, = g_train([real_images, 1])
        print ("[{0}/{1}] [{2}/{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(e, epoch, s, steps, d_loss, g_loss))

    generator.save_weights("e{0}_generator.h5".format(e))
    predict_images("e{0}_img.png".format(e), generator,noise_size,10,32)





