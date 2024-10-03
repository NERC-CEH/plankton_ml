import torch


def flat_embeddings(features: torch.Tensor) -> list:
    """Utility function that takes the features returned by the model in truncate_model
    And flattens them into a list suitable for storing in a vector database"""
    # TODO: this only returns the 0th tensor in the batch...why?
    return features[0].detach().tolist()
