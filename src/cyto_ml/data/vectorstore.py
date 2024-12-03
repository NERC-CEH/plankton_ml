import logging
import os
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import chromadb
import chromadb.api.models.Collection
from chromadb.config import Settings
from chromadb.errors import UniqueConstraintError

logging.basicConfig(level=logging.INFO)
# TODO make this sensibly configurable, not confusingly hardcoded
STORE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../vectors")


class VectorStore(metaclass=ABCMeta):
    @abstractmethod
    def add(self, url: str, embeddings: List[float]) -> None:
        pass

    @abstractmethod
    def get(self, url: str) -> List[float]:
        pass


class ChromadbStore(VectorStore):
    client = chromadb.PersistentClient(
        path=STORE,
        settings=Settings(
            anonymized_telemetry=False,
        ),
    )

    def __init__(self, db_name: str):
        try:
            collection = self.client.create_collection(
                name=db_name,
                metadata={"hnsw:space": "cosine"},  # default similarity
            )
        except UniqueConstraintError as err:
            collection = self.client.get_collection(db_name)
            logging.info(err)

        self.store = collection

    def add(self, url: str, embeddings: List[float]) -> None:
        """Add vector to Chromadb"""

        self.store.add(
            documents=[url],  # we use image location in s3 rather than text content
            embeddings=[embeddings],  # wants a list of lists
            ids=[url],  # wants a list of ids
        )

    def get(self, url: str) -> list:
        """Retrieve vector from Chromadb"""
        record = self.store.get([url], include=["embeddings"])
        return record["embeddings"][0]

    def closest(self, embeddings: list, n_results: int = 25) -> List:
        """Get the N closest identifiers by cosine distance"""
        results = self.store.query(query_embeddings=[embeddings], n_results=n_results)
        return results["ids"][0]  # by index because API assumes query always multiple inputs


class PostgresStore(VectorStore):
    def __init__(self, db_name: str):
        self.db_name = db_name

    def add(self, url: str, embeddings: List[float]) -> None:
        # Implementation for adding vector to Postgres
        pass

    def get(self, url: str) -> List[float]:
        # Implementation for retrieving vector from Postgres
        pass

    def closest(self, embeddings: list, n_results: int = 25) -> List:
        pass


class SQLiteVecStore(VectorStore):
    def __init__(self, db_name: str):
        self.db_name = db_name

    def add(self, url: str, embeddings: List[float]) -> None:
        # Implementation for adding vector to SQLite-vec
        pass

    def get(self, url: str) -> List[float]:
        # Implementation for retrieving vector from SQLite-vec
        pass

    def closest(self, embeddings: List[float], n_results: int = 25) -> List:
        pass


def vector_store(store_type: Optional[str] = "chromadb", db_name: Optional[str] = "test_collection") -> VectorStore:
    if store_type == "chromadb":
        return ChromadbStore(db_name)
    elif store_type == "postgres":
        return PostgresStore(db_name)
    elif store_type == "sqlite-vec":
        return SQLiteVecStore(db_name)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
