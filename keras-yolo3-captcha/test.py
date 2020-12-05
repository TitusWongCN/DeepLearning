import sys
import argparse
from yolo import YOLO
from PIL import Image
import requests
import time
import random

def detect_img(yolo):
    image_url = 'https://eapply.abchina.com/coin/Helper/ValidCode.ashx?0.5805915363836303'
    img = 'temp.png'
    while True:
        # input
        # img = input('Input image filename:')
        # online download
        # data = requests.get(image_url).content
        # with open(img, 'wb') as f:
        #     f.write(data)
        # random choice
        img = r'F:\WorkDir.Main\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\images\3{}.png'.format(random.randint(0, 100))
        print(img)
        try:
            image = Image.open(img, 'r')
            image.show()
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, result = yolo.detect_image(image)
            print('*************  Result is {}  *************'.format(result))
            time.sleep(5)
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
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
