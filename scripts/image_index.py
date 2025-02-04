"""Create a basic index for the images in an s3 collection"""

import yaml
import logging
import os
import s3fs
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()


def bucket_listing(bucket_name: str):
    """Create an index of images in a bucket,"""
    s3 = s3fs.S3FileSystem(anon=True, endpoint_url=os.environ["AWS_URL_ENDPOINT"])

    # This returns paths with bucket name pre-pended
    # E.g. 'untagged-images-lana/MicrobialMethane_MESO_Tank7_54.0143_-2.7770_18052023_1_17085.tif'
    contents = s3.ls(bucket_name)

    return [i.split("/")[-1] for i in filter(lambda x: x.endswith("tif"), contents)]


if __name__ == "__main__":
    # Expects the bucket name set as "collection" in params.yml (used by DVC)

    bucket_name = yaml.safe_load(open("params.yaml"))["collection"]

    images = bucket_listing(bucket_name)
    # Increment - keep a flat file index locally
    with open(f"{bucket_name}.csv", "w") as out:
        out.write("\n".join(images))
