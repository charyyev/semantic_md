import os
import pickle

import segmentation_models_pytorch as smp
import torch
import torchvision

from models.unet import Unet
from models.specialized_networks import semantic_convolution, onehot, concat, border, model_utils, simplified_onehot


# https://smp.readthedocs.io/en/latest/models.html#unet
# https://smp.readthedocs.io/en/latest/encoders.html

class ModelFactory:
    def __init__(self):
        """
        Each entry should have 4 attributes:
        (1) A function for the model contructor
        (2) kwargs for function (1)
        (3) A name under which to find pretrained weights
        (4) A model type description which specifies which the type of model. Used when extending the #channels
        """
        self.basic_models = dict(
            # unet=(Unet, {}, "unet"),
            uresnet34=(
                smp.Unet, {"encoder_name": "tu-resnet34", "encoder_weights": None, "activation": "sigmoid"},
                "resnet34", "timm_smp_res"),
            uresnet50=(
                smp.Unet, {"encoder_name": "tu-resnet50", "encoder_weights": None, "activation": "sigmoid"},
                "resnet50", "timm_smp_res"),
            uefficientnet_b2=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b2", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b2", "timm_smp_eff"),
            uefficientnet_b3=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b3", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b3", "timm_smp_eff"),
            uefficientnet_b4=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b4", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b4", "timm_smp_eff")
        )

    def get_model(self, model_type: str, pretrained_weights_path: str, config, in_channels: int = 3):
        """
        Instantiates model from available pool
        :param model_type: can be one of ['unet', 'uresnet34', 'uresnet50', 'uefficientnet_b2', 'uefficientnet_b3',
        'uefficientnet_b4', 'uefficientnet_b5', 'deepresnet34', 'deepresnet50', 'deepefficientnet_b2',
        'deepefficientnet_b3', 'deepefficientnet_b4', 'deepefficientnet_b5']
        :param in_channels: number of input channels for the initial layer
        :param pretrained_weights_path: path where pretrained weights are stored
        :param classes: number of classes to be predicted
        :param semantic_convolution: whether to use semantic map as input and conolve it
        :return: return the specified model (torch.nn.Module), as well as the pretrained transforms
        """

        if model_type == "unet":
            model = Unet(in_c=in_channels)
            transforms = {"mean": (0, 0, 0), "std": (1, 1, 1)}
            type_desc = "unet"
        elif model_type in list(self.basic_models.keys()):
            model_func, kwargs, name, type_desc = self.basic_models[model_type]

            # if it is a smp model with timm encoder, it can only handle pretrained weights with 3 input channels
            # therefore we have to manually extend the input_channels via the below function
            if type_desc in ["timm_smp_res", "timm_smp_eff"] and in_channels > 3:
                model = model_func(**kwargs, in_channels=3, classes=1)
            else:
                model = model_func(**kwargs, in_channels=in_channels, classes=1)

            metadata_path = os.path.join(pretrained_weights_path, name, "weights_object.pickle")
            with open(metadata_path, "rb") as file:
                transforms = pickle.load(file)
            weights_path = os.path.join(pretrained_weights_path, name, "weights.pth")
            weights_dict = torch.load(weights_path)
            # we have additional weights in the saved weights (for classification and stuff), so we do strict = False
            model.encoder.model.load_state_dict(weights_dict, strict=False)

            # if it is a smp model with timm encoder, it can only handle pretrained weights with 3 input channels
            # therefore we have to manually extend the input_channels via the below function
            # rarely used practice
            if type_desc in ["timm_smp_res", "timm_smp_eff"] and in_channels > 3:
                get_func, set_func = model_utils.get_set_conv1_functions(type_desc)
                add_out_channels = in_channels - 3
                model = model_utils.extend_first_convolution(model, add_out_channels, get_func, set_func)
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        get_func, set_func = model_utils.get_set_conv1_functions(type_desc)
        data_flags = config["data_flags"]
        if data_flags["semantic_convolution"]:
            model = semantic_convolution.SemanticConvolutionModel(model, in_channels, get_func, set_func)
        elif data_flags["onehot"]:
            model = onehot.OneHotModel(model, data_flags["seg_classes"], get_func, set_func)
        elif data_flags["concat"]:
            model = concat.ConcatModel(model, get_func, set_func)
        elif data_flags["border"]:
            model = border.BorderModel(model, get_func, set_func)
        elif data_flags["simplified_onehot"]:
            model = simplified_onehot.SimplifiedOneHotModel(model, get_func, set_func)

        return model, transforms
