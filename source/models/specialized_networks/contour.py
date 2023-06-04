from torch import nn

from models.specialized_networks import model_utils


class ContourModel(nn.Module):
    """
    Adds an additional input channel to receive semantic contour map
    """

    def __init__(self, pretrained_model, get_func, set_func):
        super().__init__()
        self.model = model_utils.extend_first_convolution(
            pretrained_model, 1, get_func, set_func
        )

    def forward(self, image_border):
        out = self.model(image_border)
        return out
