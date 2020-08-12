import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

content_url_input = input("Image Url of Content?")
style_url_input = input("Image Url of Sytle?")

file_name_input = input("Output file name?")
file_name = file_name_input + ".png"

output_img_size = 512
style_img_size = 256 # The Module we loaded work best for 256 pixels

content_img_size = (output_img_size, output_img_size)
style_img_size = (style_img_size, style_img_size)


def crop_center(image):
  # Returns a cropped square image.
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_img(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

# Convert Tensor to Image object so we can save the image
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

content_image = load_img(content_url_input, content_img_size)
style_image = load_img(style_url_input, style_img_size)

# Apply Avg / Max Pooling to style image (Depends on what you want)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

content_img_backup = tensor_to_image(content_image)
c_width, c_height = content_img_backup.size
print(c_width, c_height)
# Crop content image
croppedIm1 = content_img_backup.crop((0, 0, int(c_width/2), c_height))

# Fast arbitrary image style transfer
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
output_image = outputs[0]

output_image = tensor_to_image(output_image)
output_image.save("styled.png")

r_width, r_height = output_image.size
print(r_width, r_height)
croppedIm2 = output_image.crop((int(r_width/2)+1, 0, r_width, r_height))

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


output_c_image = get_concat_h(croppedIm1, croppedIm2)
output_c_image.save(file_name)