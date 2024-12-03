from cyto_ml.data.vectorstore import vector_store

import numpy as np


def test_chroma_client_no_telemetry():
    store = vector_store('chromadb')
    assert not store.client.get_settings()["anonymized_telemetry"]


def test_store():
    store = vector_store()  # default 'test_collection'
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,  # we use image location in s3 rather than text content
        embeddings=list(np.random.rand(2048)),  # wants a list of lists
    )  # wants a list of ids

    record = store.get(filename)
    assert len(record) == 2048

