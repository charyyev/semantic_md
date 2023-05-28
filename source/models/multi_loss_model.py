import os
import pickle

import torch
from torch import nn

import segmentation_models_pytorch as smp
from models.specialized_networks import model_utils
from segmentation_models_pytorch.base import SegmentationHead

smp.DeepLabV3
smp.Unet


class MultiLossModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        super_model = config["super_model"]

        self.encoder = model_utils.get_encoder(super_model)

        self.decoder_depth = model_utils.get_decoder(
            super_model, encoder_channels=self.encoder.out_channels
        )

        self.decoder_semantic = model_utils.get_decoder(
            super_model, encoder_channels=self.encoder.out_channels
        )

        self.head_depth = model_utils.get_head(
            super_model,
            in_channels=self.decoder_depth.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=3,
        )

        self.head_semantic = model_utils.get_head(
            super_model,
            in_channels=self.decoder_semantic.out_channels,
            out_channels=self.config["data_flags"]["parameters"]["seg_classes"],
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        # self.test(x)
        x = self.encoder(x)
        # if not isinstance(x, list):
        #     x = (x,)
        pred_depth = self.decoder_depth(*x)
        pred_depth = self.head_depth(pred_depth)
        pred_semantic = self.decoder_semantic(*x)
        pred_semantic = self.head_semantic(pred_semantic)
        return pred_depth, pred_semantic

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
        # self.encoder.load_state_dict(weights_dict, strict=False)

        return transforms
