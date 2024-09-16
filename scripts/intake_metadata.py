"""Heavy-handed approach to create image metadata in usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

import os
from cyto_ml.data.intake import intake_yaml
from cyto_ml.data.s3 import boto3_client, image_index


if __name__ == "__main__":

    # TODO this is a minimal change to only reflect the Lancaster data
    # Need looking harder at the Wallingford data to decide how to treat it
    # They're really distinct datasets, any benefit to sharing an index?
    image_bucket = "untagged-images-lana"

    metadata = image_index(image_bucket)

    # Pause for thought, in a better workflow we'd use the API for this?
    # Another pause for thought, should we drop intake support right now?
    s3 = boto3_client()

    catalog_csv = metadata.to_csv(index=False)
    s3.put_object(Bucket=image_bucket, Key="catalog.csv", Body=catalog_csv)

    # Write the YAML document that points to the catalog listing,
    # and a single test image
    # TODO consider just dropping all this, using DVC plus the object store API
    cat_url = f"{os.environ['AWS_URL_ENDPOINT']}/{image_bucket}/catalog.csv"
    cat_test = (
        f"{os.environ['AWS_URL_ENDPOINT']}/untagged-images-lana/19_10_Tank22_1.tif"
    )

    yaml_doc = intake_yaml(cat_test, cat_url)

    s3.put_object(Bucket=image_bucket, Key="intake.yml", Body=yaml_doc)
