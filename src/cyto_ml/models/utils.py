import torch
import torchvision

# Definitions are from here
# https://github.com/alan-turing-institute/ViT-LASNet/blob/main/test/test.py
# TODO keep these elsewhere than `utils`
# TODO consider adding the transformer model, focus on the 3 class resnet18 for now


def resnet18(num_classes: int, filename: str = "", strip_final_layer: bool = False) -> torchvision.Module:
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model_state_dict = torch.load(filename, map_location="cpu")
    model.load_state_dict(model_state_dict)

    # Return embeddings rather than the labels
    if strip_final_layer:
        model.fc = torch.nn.Identity()

    model.eval()
    return model


def flat_embeddings(features: torch.Tensor) -> list:
    """Utility function that takes the features returned by the model in truncate_model
    And flattens them into a list suitable for storing in a vector database"""
    # TODO: this only returns the 0th tensor in the batch...why?
    return features[0].detach().tolist()
