import os
import pickle

import segmentation_models_pytorch as smp
import torch

from models.unet import Unet


# https://smp.readthedocs.io/en/latest/models.html#unet
# https://smp.readthedocs.io/en/latest/encoders.html

class ModelFactory:
    def __init__(self):
        self.models = dict(
            # unet=(Unet, {}, "unet"),
            uresnet34=(
            smp.Unet, {"encoder_name": "tu-resnet34", "encoder_weights": None, "activation": "sigmoid"}, "resnet34"),
            uresnet50=(
            smp.Unet, {"encoder_name": "tu-resnet50", "encoder_weights": None, "activation": "sigmoid"}, "resnet50"),
            uefficientnet_b2=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b2", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b2"),
            uefficientnet_b3=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b3", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b3"),
            uefficientnet_b4=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b4", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b4")
        )

    def get_model(self, model_type: str, pretrained_weights_path: str, in_channels: int = 3, classes: int = 1):
        """
        Instantiates model from available pool
        :param model_type: can be one of ['unet', 'uresnet34', 'uresnet50', 'uefficientnet_b2', 'uefficientnet_b3',
        'uefficientnet_b4', 'uefficientnet_b5', 'deepresnet34', 'deepresnet50', 'deepefficientnet_b2',
        'deepefficientnet_b3', 'deepefficientnet_b4', 'deepefficientnet_b5']
        :param in_channels: number of input channels for the initial layer
        :param pretrained_weights_path: path where pretrained weights are stored
        :param classes: number of classes to be predicted
        :return: return the specified model (torch.nn.Module), as well as the pretrained transforms
        """

        if model_type == "unet":
            model = Unet(in_c=in_channels)
            transforms = {"mean": (0, 0, 0), "std": (1, 1, 1)}
        else:
            model_func, kwargs, name = self.models[model_type]
            model = model_func(**kwargs, in_channels=in_channels, classes=classes)

            metadata_path = os.path.join(pretrained_weights_path, name, "weights_object.pickle")
            with open(metadata_path, "rb") as file:
                pickled = pickle.load(file)
                transforms = pickled
            weights_path = os.path.join(pretrained_weights_path, name, "weights.pth")
            weights_dict = torch.load(weights_path)
            # we have additional weights in the saved weights (for classification and stuff), so we do strict = False
            model.encoder.model.load_state_dict(weights_dict, strict=False)

        return model, transforms
