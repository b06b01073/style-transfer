from torchvision import transforms
from torchvision.utils import save_image
import torch

from PIL import Image
from collections import defaultdict
import os

from vgg19 import VGG19

def main():
    model = VGG19()

    with torch.no_grad():
        content_representation, style_representation = get_representations(model)

    
    

def get_representations(model, visualize=False):
    '''precompute the data of Gram matrix in style representation and the content representation 
    '''
    content_img = read_img('images/content_images/cat.jpg')
    content_img_tensor = transform_img(content_img)

    content_representation = model(content_img_tensor)


    style_img = read_img('images/style_images/water_painting.jpg')
    style_img_tensor = transform_img(style_img)
    style_representation = model(style_img_tensor)
    style_representation = flatten_representation(style_representation)
    style_representation = get_gram_matrices(style_representation)

    if visualize:
        visualize_filter(style_representation, 'images/style_representation')

    content_representation = flatten_representation(content_representation)

    return content_representation, style_representation
    

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

def visualize_filter(representation, path):
    for k, v in representation.items():
        save_image(v.unsqueeze(0), os.path.join(path, f'{k}.png'))

def get_gram_matrices(style_representations):
    for layer_name, filter in style_representations.items():
        filter_trans = torch.transpose(filter, 0, 1)
        style_representations[layer_name] = torch.matmul(filter, filter_trans)
    return style_representations

def flatten_representation(representation):
    for layer_name, filter in representation.items():
        representation[layer_name] = torch.flatten(filter, start_dim=1)
    return representation

if __name__ == '__main__':
    main()