"""Thin wrapper around the s3 object store with images and metadata"""

import os

import pandas as pd
import s3fs
from dotenv import load_dotenv

load_dotenv()


def s3_endpoint() -> s3fs.S3FileSystem:
    """Return a reference to the object store,
    reading the credentials set in the environment.
    """
    fs = s3fs.S3FileSystem(
        anon=False,
        key=os.environ.get("FSSPEC_S3_KEY", ""),
        secret=os.environ.get("FSSPEC_S3_SECRET", ""),
        client_kwargs={"endpoint_url": os.environ["ENDPOINT"]},
    )
    return fs


def image_index(endpoint: s3fs.S3FileSystem, location: str) -> pd.DataFrame:
    """Find and likely later filter records in a bucket"""
    index = endpoint.ls(location)
    return pd.DataFrame(
        [f"{os.environ['ENDPOINT']}/{x}" for x in index],
        columns=["Filename"],
    )
