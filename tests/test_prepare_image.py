# test_prepare_image.py
import pytest
import torch
from cyto_ml.data.image import load_image

# https://github.com/intake/intake-xarray/blob/d0418f787181d638629b76c2982a9a215a3697be/intake_xarray/image.py#L323


def test_single_image(single_image):
    # Tensorise the image (potentially normalise if we have useful values)
    prepared_image = load_image(single_image)
    # Check if the shape is correct (batch dimension added)
    assert prepared_image.shape == torch.Size([1, 3, 80, 79])

    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))

    # Test resizing of images to 256*256
    prepared_image = load_image(single_image, normalise_func="resize_normalise")
    assert prepared_image.shape == torch.Size([1, 3, 256, 256])
    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))


def test_greyscale_image(greyscale_image):
    # Tensorise the image (potentially normalise if we have useful values)
    prepared_image = load_image(greyscale_image)
    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))


if __name__ == "__main__":
    pytest.main()
