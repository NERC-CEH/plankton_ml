from cyto_ml.data.vectorstore import vector_store, client, embeddings

import numpy as np


def test_client_no_telemetry():
    assert not client.get_settings()["anonymized_telemetry"]


def test_store():
    store = vector_store()  # default 'test_collection'
    id = "id_1"  # insists on a str
    filename = "https://example.com/filename.tif"
    store.add(
        documents=[filename],  # we use image location in s3 rather than text content
        embeddings=[list(np.random.rand(2048))],  # wants a list of lists
        ids=[id],
    )  # wants a list of ids

    record = store.get("id_1", include=["embeddings"])
    assert record
    assert len(record["embeddings"][0]) == 2048


def test_embeddings():
    store = vector_store()
    assert len(embeddings(store))
