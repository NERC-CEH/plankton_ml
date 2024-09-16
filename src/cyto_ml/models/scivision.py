import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from xarray import DataArray


def prepare_image(image: DataArray) -> torch.Tensor:
    """
    Take an xarray of image data and prepare it to pass through the model
    a) Converts the image data to a PyTorch tensor
    b) Accepts a single image or batch (no need for torch.stack)
    """
    # Computes the DataArray and returns a numpy array
    image_numpy = image.to_numpy()

    # Convert the image data to a PyTorch tensor
    tensor_image = to_dtype(
        to_image(image_numpy),  # permutes HWC -> CHW
        torch.float32,
        scale=True,  # rescales [0, 255] -> [0, 1]
    )
    assert torch.all((tensor_image >= 0.0) & (tensor_image <= 1.0))

    if tensor_image.dim() == 3:
        # Single image, add a batch dimension
        tensor_image = tensor_image.unsqueeze(0)

    assert tensor_image.dim() == 4

    return tensor_image


def flat_embeddings(features: torch.Tensor) -> list:
    """Utility function that takes the features returned by the model in truncate_model
    And flattens them into a list suitable for storing in a vector database"""
    # TODO: this only returns the 0th tensor in the batch...why?
    return features[0].detach().tolist()
