"""Heavy-handed approach to create image metadata in usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

import os
from cyto_ml.data.s3 import boto3_client, image_index


if __name__ == "__main__":

    # Write a minimal CSV index of images in a bucket
    # Was originally part of an intake catalogue setup
    image_bucket = "untagged-images-lana"

    metadata = image_index(image_bucket)

    s3 = boto3_client()

    catalog_csv = metadata.to_csv(index=False)
    s3.put_object(Bucket=image_bucket, Key="catalog.csv", Body=catalog_csv)