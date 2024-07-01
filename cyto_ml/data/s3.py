"""Thin wrapper around the s3 object store with images and metadata"""

import s3fs
from dotenv import load_dotenv
import os

load_dotenv()


def s3_endpoint():
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
