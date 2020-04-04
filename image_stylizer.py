import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import random
import string

content_url_input = input("Image Url of Content?")
style_url_input = input("Image Url of Sytle?")

# Downloads a file from a URL if it not already in the cache.
content_path = tf.keras.utils.get_file(randomFileName(10), content_url_input)
style_path = tf.keras.utils.get_file(randomFileName(11), style_url_input)

file_name_input = input("Output file name?")
file_name = file_name_input + ".png"

def randomFileName(stringLength=10):
  letters = string.ascii_lowercase
  return (''.join(random.choice(letters) for i in range(stringLength))) + ".jpg"

def load_img(path_to_img, max_dim = 1024):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

content_image = load_img(content_path)
style_image = load_img(style_path)

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# Fast arbitrary image style transfer
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image).save(file_name)