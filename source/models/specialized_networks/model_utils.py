from collections.abc import Callable

from torch import nn

from segmentation_models_pytorch import encoders
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3Decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def extend_first_convolution(
    pretrained_model: nn.Module,
    additional_out_channels: int,
    get_conv1_func: Callable,
    set_conv1_func: Callable,
):
    """
    This function extends to initial convolution of a model to more input channels,
    while keeping the pretrained weights for the non-new parts of the convolution.
    :param pretrained_model: The pretrained model to be extended
    :param additional_out_channels:  Number of channels by which the convolution should be extended
    :param get_conv1_func: A function that allows access to the first convolution via get_conv1_func(pretrained_model)
    :param set_conv1_func: A function that allows the setting of said first convolution via
    set_conv1_func(pretrained_model, new_conv1). Necessary as accessing the first conv is different for every model
    :return:
    """
    # Take the weights of the first convolutional layer of the pre-trained model
    pretrained_conv1 = get_conv1_func(pretrained_model)

    # Initialize only the weights of the first 3 channels with the pre-trained weights
    # Create exact same convolution
    in_channels = pretrained_conv1.in_channels + additional_out_channels
    own_conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=pretrained_conv1.out_channels,
        kernel_size=pretrained_conv1.kernel_size,
        stride=pretrained_conv1.stride,
        padding=pretrained_conv1.padding,
        bias=pretrained_conv1.bias,
    )
    # Create a new convolution where part of the weights are pretrained
    new_weights = own_conv1.weight.clone()
    new_weights[:, : pretrained_conv1.in_channels, :, :] = pretrained_conv1.weight[
        :, : pretrained_conv1.in_channels, :, :
    ]
    own_conv1.weight = nn.Parameter(new_weights)

    # Implant new convolution with partially pretrained weights into model
    pretrained_model = set_conv1_func(pretrained_model, own_conv1)
    return pretrained_model


# The below function return getter and setter methods for the respective models
def resnet_get_conv1_func(pretrained_model):
    return pretrained_model.encoder.model.conv1


def resnet_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv1 = conv1
    return pretrained_model


def efficientnet_get_conv1_func(pretrained_model):
    return pretrained_model.encoder.model.conv_stem


def efficientnet_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv_stem = conv1
    return pretrained_model


def deeplab_get_conv1_func(pretrained_model):
    return pretrained_model.encoder.model.conv_stem


def deeplab_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv_stem = conv1
    return pretrained_model


def unet_get_conv1_func(pretrained_model):
    return pretrained_model.encoder.model.conv1


def unet_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv1 = conv1
    return pretrained_model


def get_set_conv1_functions(type_desc):
    """
    Return the getter and setter function for the specified model type.
    """
    if type_desc == "timm_smp_res":
        return resnet_get_conv1_func, resnet_set_conv1_func
    elif type_desc == "timm_smp_eff":
        return efficientnet_get_conv1_func, efficientnet_set_conv1_func
    elif type_desc == "timm_smp_deeplab_eff":
        return deeplab_get_conv1_func, deeplab_set_conv1_func
    elif type_desc == "unet":
        return unet_get_conv1_func, unet_set_conv1_func
    else:
        raise ValueError(f"Unknown type_desc '{type_desc}'")


def get_decoder(model_description, **kwargs):
    """
    Use this function to get a decoder depending on the super_model (category 3)
    """
    if model_description == "UNet":
        return UnetDecoder(
            encoder_channels=kwargs["encoder_channels"],
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
    elif model_description == "DeepLabV3":
        return DeepLabV3Decoder(
            in_channels=kwargs["encoder_channels"][-1],
            out_channels=256,
            atrous_rates=(12, 24, 36),
        )
    else:
        raise ValueError(f"Unknown model_description '{model_description}'")


def get_encoder(model_description):
    """
    Use this function to get a encoder depending on the super_model (category 3)
    """
    if model_description == "UNet":
        return encoders.get_encoder(
            name="tu-efficientnet_b4",
            in_channels=3,
            depth=5,
            weights=None,
        )
    elif model_description == "DeepLabV3":
        return encoders.get_encoder(
            name="tu-efficientnet_b4",
            in_channels=3,
            depth=5,
            weights=None,
            output_stride=8,
        )
    else:
        raise ValueError(f"Unknown model_description '{model_description}'")


def get_head(model_description, **kwargs):
    """
    Use this function to get classification / regression head depending on the
    super_model (category 3)
    """
    if model_description == "UNet":
        return SegmentationHead(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            activation=kwargs["activation"],
            kernel_size=kwargs["kernel_size"],
            upsampling=1,
        )
    elif model_description == "DeepLabV3":
        return SegmentationHead(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            activation=kwargs["activation"],
            kernel_size=kwargs["kernel_size"],
            upsampling=8,
        )
    else:
        raise ValueError(f"Unknown model_description '{model_description}'")
