# Experimental API for serving a range of plankton models
# * Choose model endpoint
# * POST image contents
# * Probably parameter for a normalisation function with a sensible default
# * Get back a dict with classification
# * Option for confidence levels (if our models are calibrated)
# * Option to also return embeddings (could be enabled by default)
import os

import torch
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from resnet50_cefas import load_model
from torchvision import Module

from cyto_ml.data.image import load_image_from_url
from cyto_ml.data.labels import RESNET18_LABELS
from cyto_ml.models.utils import flat_embeddings, resnet18

STATE_FILE = "../../../data/weights/ResNet_18_3classes_RGB.pth"


# TODO pass state in a reproducible way - weights are on Google Drive
# TODO is there a cache decorator like streamlit's for this
def resnet18_model(num_classes: int = 3, strip_final_layer: bool = False) -> Module:
    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), STATE_FILE)
    return resnet18(num_classes=num_classes, filename=state_file, strip_final_layer=strip_final_layer)


app = FastAPI()

# TODO cache models rather than load every time


@app.get("/")
async def root() -> JSONResponse:
    return {"message": "Hello World"}


# interfaces for each of the models


@app.post("/resnet50/")
async def resnet50(url: str = Form(...)) -> JSONResponse:
    # strip_final_layer is only if we want embeddings
    model = load_model(strip_final_layer=True)
    features = model(load_image_from_url(url))
    embeddings = flat_embeddings(features)
    return {"embeddings": embeddings}


@app.post("/resnet18/")
async def resnet18_3(url: str = Form(...)) -> JSONResponse:
    # Use the 3 class Resnet18 model to return a prediction
    # TODO load into session state in a correct way
    model = resnet18_model()
    image = load_image_from_url(url)
    # TODO look at the normalisation / resize functions in Vit-lasnet tests, use them?
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # See TODOs above about loading these once
    embeddings_model = resnet18_model(strip_final_layer=True)
    outputs = embeddings_model(image)
    embeddings = flat_embeddings(outputs)

    return {"classification": RESNET18_LABELS[predicted], "embeddings": embeddings}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
