from torch import nn

class StyleTransferLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, input_representation, content_representation, style_representation=None):
        loss = self.mse_loss(input_representation['conv4_1'], content_representation['conv4_1'])
        return loss