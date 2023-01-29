from torch import nn
import torch
from collections import defaultdict

class StyleTransferLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def gram_matrices(self, style_representations):
        gram_matrices_representation = defaultdict()
        for layer_name, filter in style_representations.items():
            filter_trans = torch.transpose(filter, 0, 1)
            gram_matrices_representation[layer_name] = torch.matmul(filter, filter_trans)
        return gram_matrices_representation

    def flatten_representation(self, representation):
        flattened_representation = defaultdict()
        for layer_name, filter in representation.items():
            flattened_representation[layer_name] = torch.flatten(filter, start_dim=1)
        return flattened_representation

    def forward(self, input_representation, content_representation, style_representation):
        loss = self.alpha * self.mse_loss(input_representation['conv4_1'], content_representation['conv4_1'])
        input_grams = self.gram_matrices(self.flatten_representation(input_representation))
        style_grams = self.gram_matrices(self.flatten_representation(style_representation))

        for input_gram, style_gram in zip(input_grams.values(), style_grams.values()):
            loss += self.beta * self.mse_loss(input_gram, style_gram) / (input_gram.shape[0] ** 2) / (input_gram.shape[1] ** 2)

        return loss