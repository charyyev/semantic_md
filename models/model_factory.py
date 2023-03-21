import segmentation_models_pytorch as smp

from unet import Unet


class ModelFactory:
    def __init__(self):
        self.uresnet_kwargs = dict(encoder_name="resnet50", encoder_weights="imagenet")
        self.uefficientnet_kwargs = dict(encoder_name="efficientnet-b5", encoder_weights="imagenet")
        self.deepresnet_kwargs = dict(encoder_name="resnet50", encoder_weights="imagenet")
        self.deepefficientnet_kwargs = dict(encoder_name="efficientnet-b5", encoder_weights="imagenet")

    def get_model(self, model_type: str, in_channels: int = 3, classes: int = 3):
        if model_type == "UNet":
            return Unet(in_c=in_channels)
        elif model_type == "UResNet":
            return smp.Unet(**self.uresnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "UEfficientNet":
            return smp.Unet(**self.uefficientnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "DeepLabResNet":
            return smp.DeepLabV3Plus(**self.deepresnet_kwargs, in_channels=in_channels, classes=classes)
        elif model_type == "DeepLabEfficientNet":
            return smp.DeepLabV3Plus(**self.deepefficientnet_kwargs, in_channels=in_channels, classes=classes)
        else:
            raise ValueError(f"{model_type} is not a valid model_type.")
