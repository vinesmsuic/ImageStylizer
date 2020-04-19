import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import random
import string
import requests
from io import BytesIO

content_url_input = input("Image Url of Content?")
style_url_input = input("Image Url of Sytle?")

response = requests.get(content_url_input)
content_img_backup = Image.open(BytesIO(response.content))
#content_img_backup.save("testog.png")


c_width, c_height = content_img_backup.size
print(c_width, c_height)
# Crop content image
croppedIm1 = content_img_backup.crop((0, 0, int(c_width/2), c_height))
#croppedIm1.save("tst.png")

def randomFileName(stringLength=10):
  letters = string.ascii_lowercase
  return (''.join(random.choice(letters) for i in range(stringLength))) + ".jpg"


# Downloads a file from a URL.
content_path = tf.keras.utils.get_file(randomFileName(), content_url_input)
style_path = tf.keras.utils.get_file(randomFileName(), style_url_input)

file_name_input = input("Output file name?")
file_name = file_name_input + ".png"

'''
def load_img(path_to_img, max_dim = 512):
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
'''

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.int32)

  img = tf.image.resize(img, shape)
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
  return Image.fromarray(tensor)

# Fast arbitrary image style transfer
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
output_image = tensor_to_image(stylized_image)

output_image.save("styled.png")

r_width, r_height = output_image.size
print(r_width, r_height)
croppedIm2 = output_image.crop((int(r_width/2)+1, 0, r_width, r_height))
#croppedIm2.save("tst2.png")

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


output_c_image = get_concat_h(croppedIm1, croppedIm2)
output_c_image.save(file_name)