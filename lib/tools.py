import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from scipy import misc

def set_gpu_config(device = "0",fraction=0.25):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.visible_device_list = device
    KTF.set_session(tf.Session(config=config))


def predict_images(file_name, generator, noise_size, n = 10, size = 32):

    image = np.zeros(shape=(size*n,size*n,3))
    for i in range(0,size*n,size):
        for j in range(0,size*n,size):
            image[i:i+size, j:j+size, :] = generator.predict(np.random.normal(size=(1, ) + noise_size))[0]

    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    misc.imsave(file_name, image)