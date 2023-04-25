import os
import pickle

import torch
from torch import nn

import segmentation_models_pytorch as smp


class MultiLossModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_depth = smp.Unet(
            encoder_name="tu-efficientnet_b4",
            encoder_weights=None,
            activation="sigmoid",
        )
        model_semantic = smp.Unet(
            encoder_name="tu-efficientnet_b4",
            encoder_weights=None,
            activation=None,
        )
        self.encoder = model_depth.encoder.model
        self.decoder_depth = model_depth.decoder
        self.decoder_semantic = model_semantic.decoder

        del model_depth
        del model_semantic

    def forward(self, x):
        x = self.encoder(x)
        pred_depth = self.decoder_depth(x)
        pred_semantic = self.decoder_semantic(x)
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
        # self.encoder.model.load_state_dict(weights_dict, strict=False)
        self.encoder.load_state_dict(weights_dict, strict=False)

        return transforms
