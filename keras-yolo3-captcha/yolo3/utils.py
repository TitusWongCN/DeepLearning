"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size

    image = image.resize((w,h), Image.BICUBIC)
    new_image = Image.new('L', size)
    new_image.paste(image, (0, 0))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=4, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    image_resize = image.resize((w, h), Image.BICUBIC)
    box_data = np.zeros((max_boxes,5))
    np.random.shuffle(boxes)
    x_scale, y_scale = float(w / iw), float(h / ih)
    for index, box in enumerate(boxes):
        box[0] = int(box[0] * x_scale)
        box[1] = int(box[1] * y_scale)
        box[2] = int(box[2] * x_scale)
        box[3] = int(box[3] * y_scale)
        box_data[index, :] = box
    image_data = np.expand_dims(image_resize, axis=-1)
    image_data[image_data < 128] = 0
    image_data = np.array(image_data)/255.
    return image_data, box_data
