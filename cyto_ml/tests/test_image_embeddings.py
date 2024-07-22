from intake_xarray import ImageSource
from torch import Tensor
from cyto_ml.models.scivision import prepare_image, flat_embeddings


def test_embeddings(truncated_model, single_image):
    features = truncated_model(prepare_image(ImageSource(single_image).to_dask()))

    assert isinstance(features, Tensor)

    embeddings = flat_embeddings(features)

    assert len(embeddings) > 0
    assert len(embeddings) == features.size()[1]


def test_predictions(original_model, single_image):
    predictions = original_model(prepare_image(ImageSource(single_image).to_dask()))
    # A probably not very illuminating three output classes
    assert len(predictions.detach().cpu().numpy()[0]) == 3
