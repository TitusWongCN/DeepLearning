"""Miscellaneous utility functions."""
from functools import reduce
from PIL import Image
import numpy as np
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
    w, h = size
    image = image.resize((w,h), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, (0, 0))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, max_boxes=4):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = cv2.imread(line[0], cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    ih, iw = thresh.shape
    h, w = input_shape
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    image_resize = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_LINEAR)

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
    image_data = np.array(image_data)/255.
    return image_data, box_data

if __name__ == '__main__':
    get_random_data(r'F:\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\images\1.png 1,12,13,26,21 19,12,30,26,27 37,12,54,26,0 57,5,73,20,12', (416, 416))

