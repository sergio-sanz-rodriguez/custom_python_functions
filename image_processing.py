import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

# Function for custom normalization
def custom_normalization(image):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    image = image / 255.0
    mean = tf.constant(MEAN, dtype=image.dtype)
    std = tf.constant(STD, dtype=image.dtype)
    image = (image - mean) / std  # Normalize each channel
    return image

# Function to rotate an image by an angle
def rotate_image(image, angle):
    radians = angle * (np.pi / 180)  # Convert degrees to radians
    return tfa.image.rotate(image, radians)

# funtion to shear an image
def shear_image(image, shear, im_size):
    shear_matrix = tf.convert_to_tensor([
        [1, shear, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=tf.float32)
    image = tfa.image.transform(image, shear_matrix, interpolation='BILINEAR')
    image = tf.image.resize(image, [im_size, im_size])
    return image

# Function to zoom in or out an image
def zoom_image(image, zoom_factor, im_size):
    new_size = tf.cast([im_size * zoom_factor, im_size * zoom_factor], tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, im_size, im_size)
    return image
    
# Image augmentation
def augment_image(image, label, im_size):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    angle = tf.random.uniform([], minval=-20, maxval=20)
    image = rotate_image(image, angle)
    image = tf.image.random_crop(image, size=[im_size, im_size, 3])
    shear = tf.random.uniform([], minval=-0.2, maxval=0.2)
    image = shear_image(image, shear)
    zoom_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
    image = zoom_image(image, zoom_factor)
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label

# Parse and process images: resizing and custom normaliztion
def parse_image(filename, label, im_size):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [im_size, im_size], method=tf.image.ResizeMethod.LANCZOS3)
    image = custom_normalization(image)
    return image, label