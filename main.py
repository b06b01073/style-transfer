from vgg19 import VGG19
from PIL import Image

import os

def main():
    content_img = read_img('images/content_images/cat.jpg')

def read_img(img_path):
    dir_path = os.path.dirname(__file__) 
    img_path = os.path.join(dir_path, img_path)
    return Image.open(img_path)

if __name__ == '__main__':
    main()