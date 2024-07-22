from cyto_ml.data.intake import image_source
from xarray import DataArray


def test_image_source(single_image):
    img = image_source(single_image)
    assert isinstance(img, DataArray)
