"""Test the layout of the object store is what we expect

untagged-images-lana
untagged-images-wala
tagged-images-lana
tagged-images-wala

Inside tagged-images-lana and tagged-images-wala there is a metadata.csv file and taxonomy.csv file.

"""

import pytest
import botocore.client
from cyto_ml.data.s3 import bucket_keys, boto3_client

# Note - we skipped these tests unless running locally with credentials,
# but could develop them with moto-server ...


def test_endpoint(env_endpoint):
    if not env_endpoint:
        pytest.skip("no settings found for s3 endpoint")

    store = boto3_client()
    assert hasattr(store, "list_objects_v2")

@pytest.mark.skip("boto3 JASMIN listing issues, come back to this")
def test_img_ls(env_endpoint):
    if not env_endpoint:
        pytest.skip("no settings found for s3 endpoint")

    for bucket in ["untagged-images-lana", "untagged-images-wala"]:
        filez = bucket_keys(bucket)
        assert len([f for f in filez])
