"""Create a basic index for the images in an s3 collection
"""

import yaml
from cyto_ml.data.s3 import boto3_client, image_index


if __name__ == "__main__":

    # Write a minimal CSV index of images in a bucket
    # Was originally part of an intake catalogue setup
    image_bucket = yaml.safe_load(open("params.yaml"))["collection"]

    metadata = image_index(image_bucket)

    s3 = boto3_client()

    catalog_csv = metadata.to_csv(index=False)
    with open('catalog.csv', 'w') as out:
        out.write(catalog_csv)

    s3.put_object(Bucket=image_bucket, Key="catalog.csv", Body=catalog_csv)
