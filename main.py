from torchvision import transforms
from torchvision.utils import save_image
import torch


from PIL import Image
from collections import defaultdict
import os

from vgg19 import VGG19
from loss import StyleTransferLoss

def main():
    model = VGG19()

    content_representation, style_representation = get_representations(model)

    output_img = read_img('images/input_images/noise.jpg')
    output_img_tensor = transform_img(output_img)

    criterion = StyleTransferLoss(alpha=1, beta=1)
    optim = torch.optim.LBFGS([output_img_tensor], lr=1e-2)

    output_img_tensor.requires_grad_(True)

    criterion = torch.nn.MSELoss()


    for i in range(50):

        def closure():
            output_representation = model(output_img_tensor)
            optim.zero_grad()

                # loss = criterion(flattened_output, content_representation, style_representation)
            loss = criterion(output_representation['conv4_1'], content_representation['conv4_1'])
            # loss += torch.nn.MSELoss()(output_img_tensor, torch.zeros((3, 224, 224))) 
            loss.backward()
            print(loss)
            return loss

        optim.step(closure)

        save_image(output_img_tensor, f'images/result/{i}.jpg')


def get_representations(model, visualize=False):
    '''precompute the data of Gram matrix in style representation and the content representation 
    '''
    content_img = read_img('images/content_images/cat.jpg')
    content_img_tensor = transform_img(content_img)
    content_representation = model(content_img_tensor)



    style_img = read_img('images/style_images/water_painting.jpg')
    style_img_tensor = transform_img(style_img)
    style_representation = model(style_img_tensor)

    if visualize:
        visualize_filter(style_representation, 'images/style_representation')


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

def flatten_representation(representation):
    flattened_representation = defaultdict()
    for layer_name, filter in representation.items():
        flattened_representation[layer_name] = torch.flatten(filter, start_dim=1)
    return flattened_representation

if __name__ == '__main__':
    main()