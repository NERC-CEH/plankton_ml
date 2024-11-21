# Experimental API for serving a range of plankton models
# * Choose model endpoint
# * POST image contents
# * Probably parameter for a normalisation function with a sensible default
# * Get back a dict with classification
# * Option for confidence levels (if our models are calibrated)
# * Option to also return embeddings (could be enabled by default)
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from resnet50_cefas import load_model

from cyto_ml.data.image import load_image_from_url
from cyto_ml.models.utils import flat_embeddings

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
    return {'embeddings': embeddings}
