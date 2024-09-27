# test_prepare_image.py
import pytest
import torch
import logging
from cyto_ml.data.image import load_image

# https://github.com/intake/intake-xarray/blob/d0418f787181d638629b76c2982a9a215a3697be/intake_xarray/image.py#L323


def test_single_image(single_image):

    # Tensorise the image (potentially normalise if we have useful values)
    prepared_image = load_image(single_image)

    # Check if the shape is correct (batch dimension added)
    assert prepared_image.shape == torch.Size([1, 3, 80, 79])

    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))

if __name__ == "__main__":
    pytest.main()
