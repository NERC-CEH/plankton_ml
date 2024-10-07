import torch
from PIL import Image
from cyto_ml.models.utils import flat_embeddings
from cyto_ml.data.image import load_image, normalise_flowlr, prepare_image


def test_embeddings(resnet_model, single_image):
    features = resnet_model(load_image(single_image))

    assert isinstance(features, torch.Tensor)

    embeddings = flat_embeddings(features)

    assert len(embeddings) == features.size()[1]


def test_normalise_flowlr(greyscale_image):
    # Normalise first, hand the tensorize function an array
    image = normalise_flowlr(Image.open(greyscale_image))
    prepared_image = prepare_image(image)

    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))

    # Do it all at once
    prepared_image = prepare_image(Image.open(greyscale_image))

    assert torch.all((prepared_image >= 0.0) & (prepared_image <= 1.0))
