import os
import pickle

import segmentation_models_pytorch as smp
import torch

from models.unet import Unet


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
        self.models = dict(
            # unet=(Unet, {}, "unet"),
            uresnet34=(
                smp.Unet, {"encoder_name": "tu-resnet34", "encoder_weights": None, "activation": "sigmoid"},
                "resnet34", "timm_smp"),
            uresnet50=(
                smp.Unet, {"encoder_name": "tu-resnet50", "encoder_weights": None, "activation": "sigmoid"},
                "resnet50", "timm_smp"),
            uefficientnet_b2=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b2", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b2", "timm_smp"),
            uefficientnet_b3=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b3", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b3", "timm_smp"),
            uefficientnet_b4=(
                smp.Unet, {"encoder_name": "tu-efficientnet_b4", "encoder_weights": None, "activation": "sigmoid"},
                "efficientnet_b4", "timm_smp")
        )

    def get_model(self, model_type: str, pretrained_weights_path: str, in_channels: int = 3,
                  semantic_convolution=False):
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
        else:
            model_func, kwargs, name, type_desc = self.models[model_type]

            # if it is a smp model with timm encoder, it can only handle pretrained weights with 3 input channels
            # therefore we have to manually extend the input_channels via the below function
            if type_desc == "timm_smp" and in_channels > 3:
                model = model_func(**kwargs, in_channels=3, classes=1)
            else:
                model = model_func(**kwargs, in_channels=in_channels, classes=1)

            metadata_path = os.path.join(pretrained_weights_path, name, "weights_object.pickle")
            with open(metadata_path, "rb") as file:
                pickled = pickle.load(file)
                transforms = pickled
            weights_path = os.path.join(pretrained_weights_path, name, "weights.pth")
            weights_dict = torch.load(weights_path)
            # we have additional weights in the saved weights (for classification and stuff), so we do strict = False
            model.encoder.model.load_state_dict(weights_dict, strict=False)

            # if it is a smp model with timm encoder, it can only handle pretrained weights with 3 input channels
            # therefore we have to manually extend the input_channels via the below function
            if type_desc == "timm_smp" and in_channels > 3:
                pretrained_conv1 = model.encoder.model.conv1
                add_out_channels = in_channels - 3
                model = _extend_first_convolution(model, pretrained_conv1, _timm_set_conv1_func, add_out_channels)

        if semantic_convolution:
            model = SemanticConvolutionModel(model, semantic_out_channels=in_channels)

        return model, transforms


def _extend_first_convolution(pretrained_model, pretrained_conv1, set_conv1_func, additional_out_channels):
    """
    This function extends to initial convolution of a model to more input channels, while keeping the pretrained weights
    for the non-new parts of the convolution.
    :param pretrained_model: The pretrained model to be extended
    :param pretrained_conv1: The first convolution of the model to be extended
    :param set_conv1_func: A function that allows the setting of said first convolution via
    set_conv1_func(pretrained_model, new_conv1). Necessary as accessing the first conv is different for every model
    :param additional_out_channels:  Number of channels by which the convolution should be extended
    :return:
    """
    # Take the weights of the first convolutional layer of the pre-trained model
    pretrained_conv1 = pretrained_conv1

    # Initialize only the weights of the first 3 channels with the pre-trained weights
    # Create exact same convolution
    in_channels = pretrained_conv1.in_channels + additional_out_channels
    own_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=pretrained_conv1.out_channels,
                                kernel_size=pretrained_conv1.kernel_size, stride=pretrained_conv1.stride,
                                padding=pretrained_conv1.padding, bias=pretrained_conv1.bias)
    # Create a new convolution where part of the weights are pretrained
    new_weights = own_conv1.weight.clone()
    new_weights[:, :pretrained_conv1.in_channels, :, :] = \
        pretrained_conv1.weight[:, :pretrained_conv1.in_channels, :, :]
    own_conv1.weight = torch.nn.Parameter(new_weights)

    # Implant new convolution with partially pretrained weights into model
    pretrained_model = set_conv1_func(pretrained_model, own_conv1)
    return pretrained_model


def _timm_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv1 = conv1
    return pretrained_model


class SemanticConvolutionModel(torch.nn.Module):
    def __init__(self, pretrained_model, semantic_out_channels):
        super(SemanticConvolutionModel, self).__init__()
        self.pre_trained_model = pretrained_model
        pretrained_conv1 = self.pre_trained_model.encoder.model.conv1
        self.model = _extend_first_convolution(pretrained_model, pretrained_conv1, _timm_set_conv1_func,
                                               semantic_out_channels)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=semantic_out_channels, kernel_size=3, padding=1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, image, semantic_map):
        semantic_map_conv = self.conv(semantic_map)
        semantic_map_conv = self.sig(semantic_map_conv)
        cat = torch.cat([image, semantic_map_conv], dim=1)
        out = self.model(cat)
        return out
