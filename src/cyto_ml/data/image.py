import logging
from io import BytesIO
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms


class ImageProcessingError(Exception):
    pass


def load_image(path: str, normalise_func: Optional[str] = "base_normalise") -> torch.Tensor:
    """Given an image path, return a tensor suitable to hand to a model
    Optional normalise_func which defaults to converting to a range between 0..1
    """
    img = Image.open(path)
    return prepare_image(img, normalise_func=normalise_func)


def load_image_from_url(url: str, normalise_func: Optional[str] = "base_normalise") -> torch.Tensor:
    """Given an image url, return a tensor suitable to hand to a model
    Optional normalise_func which defaults to converting to a range between 0..1
    """
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return prepare_image(img, normalise_func=normalise_func)
    else:
        logging.error(f"{url} returned status code {response.status_code}")


def prepare_image(image: Image, normalise_func: Optional[str] = "base_normalise") -> torch.Tensor:
    """
    Take an xarray of image data and prepare it to pass through the model
    a) Converts the image data to a PyTorch tensor
    b) Accepts a single image or batch (no need for torch.stack)
    """

    if hasattr(image, "mode") and image.mode == "I;16":
        # Flow Cytometer images are 16-bit greyscale, in a low range
        # Note - tried this and variants, does not have expected result
        # https://stackoverflow.com/questions/18522295/python-pil-change-greyscale-tif-to-rgb
        #
        # Convert to 3 bands because our model has 3 channel input
        image = convert_3_band(normalise_flowlr(image))

    try:
        tensor_image = globals()[normalise_func]()(image)
    except (KeyError, Exception) as err:  # TODO trigger and catch
        logging.error(err)
        raise ImageProcessingError(err)

    # Single image, add a batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image


def base_normalise() -> transforms.Compose:
    """
    Baseline - don't standardise the values, just tensorise
    (which automatically translates to a 0-1 range)
    """
    return transforms.ToTensor()


def resize_normalise() -> transforms.Compose:
    """
    Resize to 256x256
    https://github.com/ukceh-rse/ViT-LASNet/blob/36235f9b992a6c345f1010dab133549d20f181d9/test/test.py#L115
    """
    return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def normalise_flowlr(image: Image) -> np.array:
    """Utility function to normalise flow cytometer images.
    As output from the flow cytometer, they are 16 bit greyscale,
    but all the values are in a low range (max value 1018 across the set)

    As recommended by @Kzra, normalise all values by the maximum
    Both for display, and before handing to a model.

    Image.point(lambda...) should do this, but the values stay integers
    So roundtrip this through numpy
    """
    pix = np.array(image)
    max_val = max(pix.flatten())
    pix = pix / max_val
    return pix


def convert_3_band(image: np.array) -> np.array:
    """
    Given a 1-band image normalised between 0 and 1, convert to 3 band
    https://stackoverflow.com/a/57723482
    This seems very brute-force, but PIL is not converting our odd-format
    greyscale images from the Flow Cytometer well. Improvements appreciated
    """
    img2 = np.zeros((image.shape[0], image.shape[1], 3))
    img2[:, :, 0] = image  # same value in each channel
    img2[:, :, 1] = image
    img2[:, :, 2] = image
    # Cast to float32 as this is what the model layers expect
    return img2.astype(np.float32)
