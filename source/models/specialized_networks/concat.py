from torch import nn

from source.models.specialized_networks import model_utils


class ConcatModel(nn.Module):
    def __init__(self, pretrained_model, get_func, set_func):
        super().__init__()
        self.model = model_utils.extend_first_convolution(
            pretrained_model, 1, get_func, set_func
        )

    def forward(self, image_concat):
        out = self.model(image_concat)
        return out
