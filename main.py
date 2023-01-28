from vgg19 import VGG19
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import os

def main():

    model = VGG19()

    content_img = read_img('images/content_images/cat.jpg')
    content_img_tensor = transform_img(content_img)

    content_representation = model(content_img_tensor)
    visualize_filter(content_representation, 'images/content_representation')

    style_img = read_img('images/style_images/water_painting.jpg')
    style_img_tensor = transform_img(style_img)
    style_representation = model(style_img_tensor)
    visualize_filter(style_representation, 'images/style_representation')

def read_img(img_path):
    dir_path = os.path.dirname(__file__) 
    img_path = os.path.join(dir_path, img_path)
    
    return Image.open(img_path)


def transform_img(img):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return transformer(img)

def normalize_img(img):
    return img / 255

def visualize_filter(representation, path):
    for k, v in representation.items():
        save_image(v[0].unsqueeze(0), os.path.join(path, f'{k}.png'))


if __name__ == '__main__':
    main()