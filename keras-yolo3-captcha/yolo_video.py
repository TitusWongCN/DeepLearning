import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import cv2
import numpy as np

def detect_img(yolo):
    start_index = 190
    test_dir = r'F:\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\images'
    for image_index in range(start_index, 400):
        image_file = os.path.join(test_dir, '{}.png'.format(image_index))
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        image_resize = cv2.resize(thresh, (416, 416), interpolation=cv2.INTER_LINEAR)
        image_data = np.expand_dims(image_resize, axis=-1)
        r_image, result = yolo.detect_image(image_data)
        cv2.imshow('Result: <{}>'.format(result), r_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
