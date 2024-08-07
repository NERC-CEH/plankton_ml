import os
import pytest
from cyto_ml.models.scivision import (
    load_model,
    truncate_model,
    SCIVISION_URL,
)


@pytest.fixture
def image_dir():
    """
    Existing directory of images
    """
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "fixtures/test_images/"
    )


@pytest.fixture
def single_image(image_dir):
    # The file naming conventions were like this when i got here
    return os.path.join(image_dir, "testymctestface_36.tif")


@pytest.fixture
def image_batch(image_dir):
    return os.path.join(image_dir, "testymctestface_*.tif")


@pytest.fixture
def scivision_model():
    return truncate_model(load_model(SCIVISION_URL))


@pytest.fixture
def env_endpoint():
    """None if ENDPOINT is not set in environment,
    or it's set but to an arbitrary string,
    utility for skipping integration-type tests"""
    endpoint = os.environ.get("ENDPOINT", None)
    # case in which we've got blether in the default config
    if endpoint and "https" not in endpoint:
        endpoint = None
    return endpoint
