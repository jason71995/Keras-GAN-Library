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

    image = generator.predict(np.random.normal(size=(n*n, ) + noise_size))

    image = np.reshape(image, (n, n, size, size, 3))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (n*size, n*size, 3))

    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    misc.imsave(file_name, image)