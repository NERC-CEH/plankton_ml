import logging
import os
import sqlite3
import struct
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import chromadb
import chromadb.api.models.Collection
import numpy as np
import sqlite_vec
from chromadb.config import Settings
from chromadb.errors import UniqueConstraintError

from cyto_ml.data.db_config import SQLITE_SCHEMA

logging.basicConfig(level=logging.INFO)
# TODO make this sensibly configurable, not confusingly hardcoded
STORE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../vectors")


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format
    https://github.com/asg017/sqlite-vec/blob/main/examples/simple-python/demo.py
    """
    return struct.pack("%sf" % len(vector), *vector)


def deserialize(packed: bytes) -> List[float]:
    """Inverse of the serialisation method suggested above (e.g. for clustering)"""
    size = int(len(packed) / 4)
    return struct.unpack("%sf" % size, packed)


class VectorStore(metaclass=ABCMeta):
    @abstractmethod
    def add(self, url: str, embeddings: List[float]) -> None:
        pass

    @abstractmethod
    def get(self, url: str) -> List[float]:
        pass

    @abstractmethod
    def closest(self, embeddings: List) -> List[float]:
        pass

    @abstractmethod
    def embeddings(self) -> List[List]:
        pass

    @abstractmethod
    def ids(self) -> List[str]:
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

    def closest(self, url: str, n_results: int = 25) -> List:
        """Get the N closest identifiers by cosine distance"""
        embeddings = self.get(url)
        results = self.store.query(query_embeddings=[embeddings], n_results=n_results)
        return results["ids"][0]  # by index because API assumes query always multiple inputs

    def embeddings(self) -> List[List]:
        result = self.store.get(include=["embeddings"])
        return np.array(result["embeddings"])

    def ids(self) -> List[str]:
        return self.store.get().get("ids", [])


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

    def embeddings(self) -> List[List]:
        pass

    def ids(self) -> List[str]:
        pass


class SQLiteVecStore(VectorStore):
    def __init__(self, db_name: str, embedding_len: Optional[int] = 2048, check_same_thread: bool = True):
        self._check_same_thread = check_same_thread
        self.embedding_len = embedding_len
        self.load_ext(db_name)
        self.load_schema()

    def load_ext(self, db_name: str) -> None:
        """Load the sqlite extension into our db if needed"""
        # db_name could be ':memory:' for testing, or a path
        db = sqlite3.connect(db_name, check_same_thread=self._check_same_thread)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        self.db = db

    def load_schema(self) -> None:
        """Load our db schema if needed;
        Default embedding length is 2048, set at init.
        Consider SQLAlchemy for this, or a CLI-based way of loading from a file;
        a list of CREATE TABLE statements feels like a kludge.
        """
        for statement in SQLITE_SCHEMA:
            query = statement.format(self.embedding_len)

            try:
                self.db.execute(query)
            except sqlite3.OperationalError as err:
                if "already exists" in str(err):
                    pass
                else:
                    raise

    def add(self, url: str, embeddings: List[float], classification: Optional[str] = "") -> None:
        """Add image embeddings to storage. Two tables:
        * one regular one which holds metadata, with embeddings as floats
        * one "virtual table" for indexing it by ID with encoded embeddings"""
        cursor = self.db.cursor()
        cursor.execute(
            "INSERT INTO images(url, embedding, classification) VALUES (?, ?, ?)",
            [url, serialize_f32(embeddings), classification],
        )

        row_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO images_vec(id, embedding) VALUES (?, ?)",
            [row_id, serialize_f32(embeddings)],
        )

        self.db.commit()

    def get(self, url: str) -> List[float]:
        result = self.db.execute("select embedding from images where url = ?", [url]).fetchone()
        if len(result):
            return result[0]
        else:
            return None

    def closest(self, url: str, n_results: int = 25) -> List:
        """Find and return the N closest examples by distance
        Accepts an image URL, returns a list ordered by distance
        """
        # See https://til.simonwillison.net/sqlite/sqlite-vec
        # https://github.com/asg017/sqlite-vec/issues/41 - "limit ?" not guaranteed
        # Note - stopped returning distance for consistency, but might be useful

        try:
            doc_id = self.db.execute("select id from images where url = ?", [url]).fetchone()[0]
        except IndexError:
            return None

        query = """
            with image_embedding as (
                select embedding as first_embedding from images_vec where id = ?
            )
            select
            images.id,
            images.url,
            vec_distance_cosine(images_vec.embedding, first_embedding) as distance
            from
            images_vec, image_embedding, images
            order by distance limit ?"""

        results = self.db.execute(query, [doc_id, n_results]).fetchall()
        return [i for j in results for i in j]

    def labelled(self, label: str, n_results: int = 50) -> List[str]:
        labelled = self.db.execute(
            """select url from images where classification = ? limit ?""", (label, n_results)
        ).fetchall()
        return [i for j in labelled for i in j]

    def classes(self) -> List[str]:
        classes = self.db.execute("""select distinct classification from images""").fetchall()
        return [i for j in classes for i in j]

    def embeddings(self) -> List[List]:
        embeddings = self.db.execute("""select embedding from images""").fetchall()
        return [deserialize(i) for j in embeddings for i in j]

    def ids(self) -> List[str]:
        urls = self.db.execute("""select url from images""").fetchall()
        return [i for j in urls for i in j]


def vector_store(
    store_type: Optional[str] = "chromadb", db_name: Optional[str] = "test_collection", **kwargs
) -> VectorStore:
    if store_type == "chromadb":
        return ChromadbStore(db_name, **kwargs)
    elif store_type == "postgres":
        return PostgresStore(db_name, **kwargs)
    elif store_type == "sqlite":
        return SQLiteVecStore(db_name, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
