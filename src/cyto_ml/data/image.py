import logging
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    return prepare_image(img)


def load_image_from_url(url: str) -> torch.Tensor:
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return prepare_image(img)
    else:
        logging.error(f"{url} returned status code {response.status_code}")


def prepare_image(image: Image) -> torch.Tensor:
    """
    Take an xarray of image data and prepare it to pass through the model
    a) Converts the image data to a PyTorch tensor
    b) Accepts a single image or batch (no need for torch.stack)
    """
    # Flow Cytometer images are 16-bit greyscale
    # https://stackoverflow.com/questions/18522295/python-pil-change-greyscale-tif-to-rgb
    # TODO revisit

    if image.mode == "I;16":
        image.point(lambda p: p * 0.0039063096, mode="RGB")
        image = image.convert("RGB")

    tensor_image = transforms.ToTensor()(image)

    # Single image, add a batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image
