from torchvision.models import vgg19 
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor 
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg19 = vgg19(weights='IMAGENET1K_V1').features

        self.max_pool_layers = [4, 9, 18, 27]

        for param in self.vgg19.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.vgg19):
            if hasattr(module, 'inplace'):
                self.vgg19[idx].inplace = False
            if idx in self.max_pool_layers:
                self.vgg19[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.model = create_feature_extractor(self.vgg19, {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1',
        })

    def forward(self, x):
        output = self.model(x)
        return output
