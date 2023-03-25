import segmentation_models_pytorch as smp

from models.unet import Unet

# https://smp.readthedocs.io/en/latest/models.html#unet
# https://smp.readthedocs.io/en/latest/encoders.html

class ModelFactory:
    def __init__(self):
        self.uresnet_kwargs = dict(encoder_name="resnet50", encoder_weights="imagenet")
        self.uefficientnet_kwargs = dict(encoder_name="efficientnet-b5", encoder_weights="imagenet")
        self.deepresnet_kwargs = dict(encoder_name="resnet50", encoder_weights="imagenet")
        self.deepefficientnet_kwargs = dict(encoder_name="efficientnet-b5", encoder_weights="imagenet")

    def get_model(self, model_type: str, in_channels: int = 3, classes: int = 1):
        """
        Instantiates model from available pool
        :param model_type: can be one of ['Unet', 'UResNet', 'UEfficientNet', 'DeepLabResNet', 'DeepLabEfficientNet']
        :param in_channels: number of input channels for the initial layer
        :param classes: number of classes to be predicted
        :return: return the specified model (torch.nn.Module)
        """
        # TODO: Check what each model was trained on (input range) and set flags accordingly
        # TODO: possibly return flags as additional output for dataloader

        if model_type == "UNet":
            model = Unet(in_c=in_channels)
        elif model_type == "UResNet":
            model = smp.Unet(**self.uresnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "UEfficientNet":
            model = smp.Unet(**self.uefficientnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "DeepLabResNet":
            model = smp.DeepLabV3Plus(**self.deepresnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "DeepLabEfficientNet":
            model = smp.DeepLabV3Plus(**self.deepefficientnet_kwargs, in_channels=in_channels, classes=classes)
        else:
            raise ValueError(f"{model_type} is not a valid model_type.")

        return model, None  # None for now
