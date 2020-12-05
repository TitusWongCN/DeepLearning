import os
import xml.etree.ElementTree as ET
import cv2

labels = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q',
          'R','S','T','U','V','W','X','Y','Z','2','3','4','5','6','7','8']
dirpath = r'F:\WorkDir.Main\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\labels'  # 原来存放xml文件的目录
for fp in os.listdir(dirpath):
    root = ET.parse(os.path.join(dirpath, fp)).getroot()
    filename = root.find('filename').text
    img_path = r'F:\WorkDir.Main\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\images\{}.png'.format(filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    boxes = [img_path, ]
    for child in root.findall('object'):  # 找到图片中的所有框
        label = child.find('name')
        label_index = labels.index(label.text)
        sub = child.find('bndbox')  # 找到框的标注值并进行读取
        xmin = sub[0].text
        ymin = sub[1].text
        xmax = sub[2].text
        ymax = sub[3].text
        boxes.append(','.join([xmin, ymin, xmax, ymax, str(label_index)]))
    with open(R'F:\WorkDir.Main\WorkDir\DeepLearning\keras-yolo3-captcha\data\captchas\data.txt', 'a+') as f:
        f.write(' '.join(boxes) + '\n')
