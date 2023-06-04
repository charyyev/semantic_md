from torch import nn

from models.specialized_networks import model_utils


class OneHotModel(nn.Module):
    """
    Adds additional input channels to receive onehot encoded semantic classes
    """
    def __init__(self, pretrained_model, nr_classes, get_func, set_func):
        super().__init__()
        self.model = model_utils.extend_first_convolution(
            pretrained_model, nr_classes, get_func, set_func
        )

    def forward(self, image_semantic_onehot):
        out = self.model(image_semantic_onehot)
        return out
