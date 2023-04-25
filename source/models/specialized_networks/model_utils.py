from torch import nn


def extend_first_convolution(pretrained_model, additional_out_channels, get_conv1_func, set_conv1_func):
    """
    This function extends to initial convolution of a model to more input channels, while keeping the pretrained weights
    for the non-new parts of the convolution.
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
    own_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=pretrained_conv1.out_channels,
                          kernel_size=pretrained_conv1.kernel_size, stride=pretrained_conv1.stride,
                          padding=pretrained_conv1.padding, bias=pretrained_conv1.bias)
    # Create a new convolution where part of the weights are pretrained
    new_weights = own_conv1.weight.clone()
    new_weights[:, :pretrained_conv1.in_channels, :, :] = \
        pretrained_conv1.weight[:, :pretrained_conv1.in_channels, :, :]
    own_conv1.weight = nn.Parameter(new_weights)

    # Implant new convolution with partially pretrained weights into model
    pretrained_model = set_conv1_func(pretrained_model, own_conv1)
    return pretrained_model


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


def unet_get_conv1_func(pretrained_model):
    return pretrained_model.encoder.model.conv1


def unet_set_conv1_func(pretrained_model, conv1):
    pretrained_model.encoder.model.conv1 = conv1
    return pretrained_model


def get_set_conv1_functions(type_desc):
    if type_desc == "timm_smp_res":
        return resnet_get_conv1_func, resnet_set_conv1_func
    elif type_desc == "timm_smp_eff":
        return efficientnet_get_conv1_func, efficientnet_set_conv1_func
    elif type_desc == "unet":
        return unet_get_conv1_func, unet_set_conv1_func
    else:
        raise ValueError(f"Unknown type_desc '{type_desc}'")
