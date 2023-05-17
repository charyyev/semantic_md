import os
import pickle

import torch
from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class SobelLossModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.test = smp.Unet(
            encoder_name="tu-efficientnet_b4", encoder_depth=5, encoder_weights=None
        )

        self.encoder = get_encoder(
            name="tu-efficientnet_b4",
            in_channels=3,
            depth=5,
            weights=None,
        )

        self.decoder_depth = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.decoder_sobel = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.head_depth = SegmentationHead(
            in_channels=16,
            out_channels=1,
            activation=None,
            kernel_size=3,
        )

        self.head_sobel = SegmentationHead(
            in_channels=16,
            out_channels=2,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        # self.test(x)
        x = self.encoder(x)
        pred_depth = self.decoder_depth(*x)
        pred_depth = self.head_depth(pred_depth)
        pred_sobel = self.decoder_sobel(*x)
        pred_sobel = self.head_sobel(pred_sobel)
        return pred_depth, pred_sobel

    def load_and_transforms(self):
        pretrained_weights_path = self.config.get_subpath("pretrained_weights_path")

        metadata_path = os.path.join(
            pretrained_weights_path,
            "efficientnet_b4",
            self.config["pretrained_names"]["pretrained_metadata"],
        )
        with open(metadata_path, "rb") as file:
            transforms = pickle.load(file)

        weights_path = os.path.join(
            pretrained_weights_path,
            "efficientnet_b4",
            self.config["pretrained_names"]["weights"],
        )
        weights_dict = torch.load(weights_path)
        self.encoder.model.load_state_dict(weights_dict, strict=False)

        return transforms