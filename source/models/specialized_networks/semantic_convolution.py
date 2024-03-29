import torch
import torchvision
from torch import nn

from models.specialized_networks import model_utils


class SemanticConvolutionModel(nn.Module):
    """
    Adds an input to the model, which takes the semantic segmentation map
    and feeds it through an MBConv block before concatenating it to the RGB image.
    """

    def __init__(self, pretrained_model, semantic_out_channels, get_func, set_func):
        super().__init__()
        self.pretrained_model = pretrained_model
        # extend normal model by doubling input channels
        self.model = model_utils.extend_first_convolution(
            pretrained_model, semantic_out_channels, get_func, set_func
        )

        # specify network for convolving semantic map
        mbconv_config = torchvision.models.efficientnet.MBConvConfig(
            expand_ratio=6,
            kernel=3,
            stride=1,
            input_channels=1,
            out_channels=semantic_out_channels,
            num_layers=1,
            width_mult=1.4,
            depth_mult=1.8,
        )
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=8),
            torchvision.models.efficientnet.MBConv(mbconv_config, 0.2, nn.BatchNorm2d),
            nn.Conv2d(
                in_channels=8,
                out_channels=semantic_out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=semantic_out_channels),
            nn.Sigmoid(),
        )

    def forward(self, image, semantic_map):
        # convolve semantic map
        semantic_map_conv = self.convolution(semantic_map)
        # concatenate with RGB
        cat = torch.cat([image, semantic_map_conv], dim=1)
        # feed to normal network
        out = self.model(cat)
        return out
