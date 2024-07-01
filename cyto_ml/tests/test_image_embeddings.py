from intake_xarray import ImageSource
from torch import Tensor
from cyto_ml.models.scivision import prepare_image, flat_embeddings


def test_embeddings(scivision_model, single_image):
    features = scivision_model(prepare_image(ImageSource(single_image).to_dask()))

    assert isinstance(features, Tensor)

    embeddings = flat_embeddings(features)

    assert len(embeddings) == features.size()[1]
