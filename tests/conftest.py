import os
import pytest
import shutil
from resnet50_cefas import load_model


@pytest.fixture
def fixture_dir():
    """
    Base directory for the test fixtures (images, metadata)
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "fixtures/")


@pytest.fixture
def image_dir(fixture_dir):
    """Directory with single plankton images"""
    return os.path.join(fixture_dir, "test_images")


@pytest.fixture
def single_image(image_dir):
    # The file naming conventions were like this when i got here
    return os.path.join(image_dir, "testymctestface_36.tif")


@pytest.fixture
def image_batch(image_dir):
    return os.path.join(image_dir, "testymctestface_*.tif")


@pytest.fixture
def scivision_model():
    return load_model(strip_final_layer=True)


@pytest.fixture
def env_endpoint():
    """None if ENDPOINT is not set in environment,
    or it's set but to an arbitrary string,
    utility for skipping integration-type tests"""
    endpoint = os.environ.get("AWS_URL_ENDPOINT", None)
    # case in which we've got blether in the default config
    if endpoint and "https" not in endpoint:
        endpoint = None
    return endpoint


@pytest.fixture
def lst_file(fixture_dir):
    """Location of a metadata file for a FlowCam image batch"""
    return os.path.join(
        fixture_dir,
        "MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1/metadata.lst",
    )


@pytest.fixture
def collage_file(fixture_dir):
    """Location of a collage file with a FlowCam image batch"""
    return os.path.join(
        fixture_dir,
        "MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1/MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1_images_000001.tif",  # noqa: E501
    )  # noqa: E501


@pytest.fixture
def exiftest_file(fixture_dir):
    """This runs in-place so make a copy of the file every time"""
    orig = os.path.join(
        fixture_dir,
        "MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1/exiftest.tif",
    )
    temp = orig.replace("exiftest", "temp")
    shutil.copyfile(orig, temp)
    return temp
