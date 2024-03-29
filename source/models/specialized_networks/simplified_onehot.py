from torch import nn

from models.specialized_networks import model_utils


class SimplifiedOneHotModel(nn.Module):
    """
    Adds additional input channels to receive onehot encoded semantic classes
    """

    def __init__(self, pretrained_model, get_func, set_func, num_encode):
        super().__init__()
        self.model = model_utils.extend_first_convolution(
            pretrained_model, num_encode, get_func, set_func
        )

    def forward(self, image_semantic_onehot):
        out = self.model(image_semantic_onehot)
        return out
