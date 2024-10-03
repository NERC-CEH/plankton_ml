
from torch import Tensor
from cyto_ml.models.utils import flat_embeddings
from cyto_ml.data.image import load_image


def test_embeddings(scivision_model, single_image):
    features = scivision_model(load_image(single_image))

    assert isinstance(features, Tensor)

    embeddings = flat_embeddings(features)

    assert len(embeddings) == features.size()[1]

