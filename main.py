from torchvision import transforms
from torchvision.utils import save_image
import torch

import argparse

from PIL import Image
import os
import logging

from vgg19 import VGG19
from loss import StyleTransferLoss

def main(output, content, style, iters, alpha, beta, lr, saved_dir):
    model = VGG19()

    content_representation, style_representation = get_representations(model, content, style)

    output_img = read_img(output)
    output_img_tensor = transform_img(output_img)

    criterion = StyleTransferLoss(alpha=alpha, beta=beta)
    optim = torch.optim.LBFGS([output_img_tensor], lr=lr)

    output_img_tensor.requires_grad_(True)

    for i in range(iters):

        def closure():
            output_representation = model(output_img_tensor)
            optim.zero_grad()

            loss = criterion(output_representation, content_representation, style_representation)
            # loss += torch.nn.MSELoss()(output_img_tensor, torch.zeros((3, 224, 224))) 
            loss.backward()
            logging.info(f'loss: {loss.item()}')
            return loss

        optim.step(closure)

        img_path = os.path.dirname(__file__)
        img_path = os.path.join(img_path, saved_dir, f'{i}.jpg')
        logging.info(f'{img_path} saved')
        save_image(output_img_tensor, img_path)


def get_representations(model, content, style):
    '''precompute the data of Gram matrix in style representation and the content representation 
    '''
    content_img = read_img(content)
    content_img_tensor = transform_img(content_img)
    content_representation = model(content_img_tensor)

    style_img = read_img(style)
    style_img_tensor = transform_img(style_img)
    style_representation = model(style_img_tensor)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default='images/content_images/cat.jpg')
    parser.add_argument('--content', type=str, default='images/content_images/cat.jpg')
    parser.add_argument('--style', type=str, default='images/style_images/sunrise.jpg')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1e4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--saved_dir', type=str, default='images/result')
    parser.add_argument('--quite', '-q', action='store_true')

    args = parser.parse_args()

    if not args.quite:
        logging.basicConfig(level=logging.INFO)

    main(args.output, args.content, args.style, args.iters, args.alpha, args.beta, args.lr, args.saved_dir)