from torchvision.models import vgg19 
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor 

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg19 = vgg19(weights='IMAGENET1K_V1').features
        self.model = create_feature_extractor(self.vgg19, {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1',
        })

    def forward(self, x):
        x = self.model(x)
        return x
