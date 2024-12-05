import os
from fastapi.testclient import TestClient

from cyto_ml.models.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_resnet50_endpoint():
    url = "https://cdn.oceanservice.noaa.gov/oceanserviceprod/facts/nasa-copepod.jpg"

    params = {"url": url}
    response = client.post("/resnet50/", data=params)

    assert response.status_code == 200

    doc = response.json()
    assert "embeddings" in doc


def test_resnet18_endpoint():
    url = "https://cdn.oceanservice.noaa.gov/oceanserviceprod/facts/nasa-copepod.jpg"

    params = {"url": url}
    # TODO write more detailed tests if this extends beyond a prototype

    response = client.post("/resnet18/", data=params)

    # TODO - this is horribly brittle, need a version to download
    # See https://github.com/alan-turing-institute/ViT-LASNet/issues/2
    state = "./data/weights/ResNet_18_3classes_RGB.pth"
    if os.path.exists(state):
        assert response.status_code == 200
        doc = response.json()
        assert "embeddings" in doc
    else:
        # Apparently ok to return 404 if the endpoint exists but it can't act
        assert response.status_code == 404
        doc = response.json()
        assert "error" in doc
