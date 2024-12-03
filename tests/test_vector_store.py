from cyto_ml.data.vectorstore import vector_store, STORE

import numpy as np
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory using pytest's tmp_path fixture."""
    return tmp_path


def test_chroma_client_no_telemetry(temp_dir):
    STORE = temp_dir
    store = vector_store("chromadb")
    assert not store.client.get_settings()["anonymized_telemetry"]


def test_store(temp_dir):
    STORE = temp_dir
    store = vector_store()  # default 'test_collection'
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,  # we use image location in s3 rather than text content
        embeddings=list(np.random.rand(2048)),  # wants a list of lists
    )  # wants a list of ids

    record = store.get(filename)
    assert len(record) == 2048


def test_embeddings(temp_dir):
    STORE = temp_dir
    store = vector_store("chromadb", "tmp")
    filename = "https://example.com/filename.tif"
    store.add(
        url=filename,  # we use image location in s3 rather than text content
        embeddings=list(np.random.rand(2048)),  # wants a list of lists
    )
    total = store.embeddings()
    assert len(total)


def test_closest():
    store = vector_store("chromadb", "tmp")
    for i in range(0, 5):
        filename = f"https://example.com/filename{i}.tif"
        store.add(
            url=filename,  # we use image location in s3 rather than text content
            embeddings=list(np.random.rand(2048)),  # wants a list of lists
        )

    sample = store.get("https://example.com/filename0.tif")
    close = store.closest(sample)
    assert len(close)
