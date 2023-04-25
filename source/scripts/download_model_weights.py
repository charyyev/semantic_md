import os
import pickle
import urllib

import torch

import timm

from source.utils.config import Config

_MODEL_NAMES = [
    "resnet34",
    "resnet50",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
]


def test(pretrained_weights_path):
    for name in _MODEL_NAMES:
        weights_path = os.path.join(pretrained_weights_path, name, "weights.pth")
        model = timm.create_model(name, pretrained=False)

        weights_dict = torch.load(weights_path)
        model.load_state_dict(weights_dict)

        weights_object_pickle_path = os.path.join(
            pretrained_weights_path, name, "weights_object.pickle"
        )
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
        raise FileExistsError(f"{pretrained_weights_path} does not exist.")

    for name in _MODEL_NAMES:
        current_dir = os.path.join(pretrained_weights_path, name)
        os.makedirs(current_dir, exist_ok=True)

        current_weights_path = os.path.join(current_dir, "weights.pth")
        if download_from_link:
            model = timm.create_model(name, pretrained=False)
            download_link = model.pretrained_cfg["url"]
            urllib.request.urlretrieve(download_link, current_weights_path)
        else:
            model = timm.create_model(name, pretrained=True)
            torch.save(model.state_dict(), current_weights_path)

        metadata = model.pretrained_cfg
        with open(os.path.join(current_dir, "weights_object.pickle"), "wb") as file:
            pickle.dump(metadata, file)


def main(download_from_link=False):
    config = Config()
    pretrained_weights_path = config.get_config("pretrained_weights_path")
    download(pretrained_weights_path, download_from_link)
    test(pretrained_weights_path)


if __name__ == "__main__":
    main(download_from_link=False)
