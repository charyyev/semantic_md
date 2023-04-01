import os
import pickle
import torch
from torchvision import models
import urllib

from utils.config import args_and_config

_WEIGHT_PATHS = [
    ("resnet34", models.resnet34, models.ResNet34_Weights, "https://download.pytorch.org/models/resnet34-b627a593.pth"),
    ("resnet50", models.resnet50, models.ResNet50_Weights, "https://download.pytorch.org/models/resnet50-0676ba61.pth"),
    ("efficientnet_b4", models.efficientnet_b4, models.EfficientNet_B4_Weights,
     "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"),
    ("efficientnet_b5", models.efficientnet_b5, models.EfficientNet_B5_Weights,
     "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth"),
]


def test(pretrained_weights_path):
    for name, model_type, _, _ in _WEIGHT_PATHS:
        weights_path = os.path.join(pretrained_weights_path, name, "weights.pth")
        model = model_type(weights=None)

        weights_dict = torch.load(weights_path)
        model.load_state_dict(weights_dict)

        weights_object_pickle_path = os.path.join(pretrained_weights_path, name, "weights_object.pickle")
        with open(weights_object_pickle_path, "rb") as file:
            pickled = pickle.load(file)
            print(pickled)


def download(pretrained_weights_path, download_from_link):
    """
    Download pretrained weights object (includes info like normalization parameters), along with weight values.
    For closer description, look at the README.md in models/pretrained_weights
    :param pretrained_weights_path: the folder where objects are to be stored
    :param download_from_link: whether to download the weight values directly from the internet (True), or indirectly
    via the torch library (might not work on server)
    :return:
    """
    if not os.path.isdir(pretrained_weights_path):
        raise FileExistsError(f'{pretrained_weights_path} does not exist.')

    for name, model_type, weights, download_link in _WEIGHT_PATHS:
        current_dir = os.path.join(pretrained_weights_path, name)
        os.makedirs(current_dir, exist_ok=True)

        with open(os.path.join(current_dir, "weights_object.pickle"), "wb") as file:
            pickle.dump(weights, file)

        current_weights_path = os.path.join(current_dir, "weights.pth")
        if download_from_link:
            urllib.request.urlretrieve(download_link, current_weights_path)
        else:
            model = model_type(weights=weights)
            torch.save(model.state_dict(), current_weights_path)


def main(download_from_link=False):
    config = args_and_config()
    pretrained_weights_path = os.path.join(config["root_dir"], "models", "pretrained_weights")
    download(pretrained_weights_path, download_from_link)
    test(pretrained_weights_path)


if __name__ == '__main__':
    main(download_from_link=False)
