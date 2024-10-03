import logging
import os
from typing import Optional

import chromadb
import chromadb.api.models.Collection
import numpy as np
from chromadb.config import Settings
from chromadb.db.base import UniqueConstraintError

logging.basicConfig(level=logging.INFO)
# TODO make this sensibly configurable, not confusingly hardcoded
STORE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../vectors")

client = chromadb.PersistentClient(
    path=STORE,
    settings=Settings(
        anonymized_telemetry=False,
    ),
)


def vector_store(name: Optional[str] = "test_collection") -> chromadb.api.models.Collection.Collection:
    """
    Return a vector store specified by name, default test_collection
    """
    try:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # default similarity
        )
    except UniqueConstraintError as err:
        collection = client.get_collection(name)
        logging.info(err)

    return collection


def embeddings(store: chromadb.api.models.Collection.Collection) -> np.ndarray:
    """
    Shortcut to return all the embeddings in a collection, as a list
    """
    result = store.get(include=["embeddings"])
    return np.array(result["embeddings"])
