from scivision import load_pretrained_model
from scivision.io import PretrainedModel
import torch
import torchvision
from xarray import DataArray

SCIVISION_URL = "https://github.com/alan-turing-institute/plankton-cefas-scivision"  # noqa: E501


def load_model(url: str):
    """Load a scivision model from a URL, for example
    https://github.com/alan-turing-institute/plankton-cefas-scivision
    """
    model = load_pretrained_model(url)
    return model


def truncate_model(model: PretrainedModel):
    """
    Accepts a scivision model wrapper and returns the pytorch model,
    with its last fully-connected layer removed so that it returns
    2048 features rather than a handle of label predictions
    """
    network = torch.nn.Sequential(
        *(list(model._plumbing.model.pretrained_model.children())[:-1])
    )
    return network


def prepare_image(image: DataArray):
    """
    Take an xarray of image data and prepare it to pass through the model
    a) Converts the image data to a PyTorch tensor
    b) Accepts a single image or batch (no need for torch.stack)
    c) Uses a CUDA device if available
    """
    # Convert the image data to a PyTorch tensor
    tensor_image = torchvision.transforms.ToTensor()(image.to_numpy())

    # Check if the input is a single image or a batch
    if len(tensor_image.shape) == 3:
        # Single image, add a batch dimension
        tensor_image = tensor_image.unsqueeze(0)

    # Check if CUDA is available and move the tensor to the CUDA device
    if torch.cuda.is_available():
        tensor_image = tensor_image.cuda()

    return tensor_image
