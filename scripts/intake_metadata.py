"""Heavy-handed approach to create image metadata in usable with `intake`,
for trial use with `scivision`:
https://scivision.readthedocs.io/en/latest/api.html#scivision.io.reader.load_dataset
https://intake.readthedocs.io/en/latest/catalog.html#yaml-format

See also https://github.com/intake/intake-stac
Via https://gallery.pangeo.io/repos/pangeo-data/pangeo-tutorial-gallery/intake.html#Build-an-intake-catalog

"""

from cyto_ml.data.intake import intake_yaml
from cyto_ml.data.s3 import s3_endpoint
from s3fs import S3FileSystem
import pandas as pd
import os


def image_index(endpoint: S3FileSystem, location: str):
    """Find and likely later filter records in a bucket"""
    index = endpoint.ls(location)
    return pd.DataFrame(
        [f"{os.environ['ENDPOINT']}/{x}" for x in index],
        columns=["Filename"],
    )


if __name__ == "__main__":

    fs = s3_endpoint()
    metadata = image_index(fs, "untagged-images")

    # Option to use a CSV as an index, rather than return the files
    catalog = "metadata/catalog.csv"
    with fs.open(catalog, "w") as out:
        out.write(metadata.to_csv(index=False))

    cat_url = f"{os.environ['ENDPOINT']}/{catalog}"

    with fs.open("metadata/intake.yml", "w") as out:
        # Do we use a CSV driver and include the metadata?
        # out.write(write_yaml(f"{os.environ['ENDPOINT']}/{catalog}"))

        # See the issue here: https://github.com/NERC-CEH/plankton_ml/issues/3
        # About data improvements needed before a better way to read a bucket into s3
        cat_test = f"{os.environ['ENDPOINT']}/untagged-images/19_10_Tank22_1.tif"

        # Our options for the whole collection look like:
        # * a tiny http server that creates a zip, but assumes the images have more metadata
        # * a tabular index instead, means we get less advantage from intake though

        out.write(intake_yaml(cat_test, cat_url))
