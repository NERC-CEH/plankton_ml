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

from cyto_ml.data.image import load_image_from_url
from cyto_ml.data.labels import RESNET18_LABELS
from cyto_ml.models.utils import flat_embeddings, resnet18

STATE_FILE = "../../../data/weights/ResNet_18_3classes_RGB.pth"

# 3-class ResNet18, newer work from Turing Inst
# https://noushineftekhari.github.io/publication/2024-marine-plankton-classification
resnet18_classifier = None
resnet18_embeddings = None

# Fork of earlier Turing model via sci.vision
# https://github.com/ukceh-rse/resnet50-cefas
resnet50_model = None

app = FastAPI()

# TODO look at ProcessPoolExecutor with load function for concurrent requests
# https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d
# The load_models function here is made with that in mind.


# TODO pass state in a reproducible way - weights are on Google Drive
# This could easily become overkill if we start adding ViT, BioCLIP etc
def load_models() -> None:
    global resnet18_classifier  # noqa PLW0603
    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), STATE_FILE)
    resnet18_classifier = resnet18(num_classes=3, filename=state_file)

    global resnet18_embeddings  # noqa PLW0603
    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), STATE_FILE)
    resnet18_embeddings = resnet18(num_classes=3, filename=state_file, strip_final_layer=True)

    global resnet50_model  # noqa PLW0603
    resnet50_model = load_model(strip_final_layer=True)


load_models()


@app.get("/")
async def root() -> JSONResponse:
    return {"message": "Hello World"}


# interfaces for each of the models


@app.post("/resnet50/")
async def resnet50(url: str = Form(...)) -> JSONResponse:
    # strip_final_layer is only if we want embeddings

    features = resnet50_model(load_image_from_url(url))
    embeddings = flat_embeddings(features)
    return {"embeddings": embeddings}


@app.post("/resnet18/")
async def resnet18_3(url: str = Form(...)) -> JSONResponse:
    """Use the 3 class Resnet18 model to return both a prediction
    and a set of image embeddings"""

    image = load_image_from_url(url)
    # TODO look at the normalisation / resize functions in Vit-lasnet tests, use them?
    outputs = resnet18_classifier(image)
    _, predicted = torch.max(outputs, 1)

    outputs = resnet18_embeddings(image)
    embeddings = flat_embeddings(outputs)

    return {"classification": RESNET18_LABELS[predicted], "embeddings": embeddings}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
