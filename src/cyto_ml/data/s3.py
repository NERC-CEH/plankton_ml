"""Thin wrapper around the s3 object store with images and metadata"""

import os
from typing import Generator

import boto3
import pandas as pd
from dotenv import load_dotenv

# Load standard connection details via .env
load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_URL_ENDPOINT = os.environ.get("AWS_URL_ENDPOINT", "")


def boto3_client() -> boto3.Session:
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_URL_ENDPOINT,
    )


def bucket_keys(
    bucket_name: str, prefix: str = "/", delimiter: str = "/", start_after: str = ""
) -> Generator[str, None, None]:
    """Efficiently the contents of a bucket
    Lifted from this highly-rated SO answer: https://stackoverflow.com/a/54014862"""

    s3_paginator = boto3_client().get_paginator("list_objects_v2")
    prefix = prefix.lstrip(delimiter)
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
        for content in page.get("Contents", ()):
            yield content["Key"]


def image_index(location: str, suffix: str = ".tif") -> pd.DataFrame:
    """Find records in a bucket, return a DataFrame serving as an index
    Filter by optional file suffix, which by default is .tif"""
    index = bucket_keys(location)
    index = list(filter(lambda x: suffix in x, index))
    return pd.DataFrame(
        [f"{os.environ['AWS_URL_ENDPOINT']}/{location}/{x}" for x in index],
        columns=["Filename"],
    )
