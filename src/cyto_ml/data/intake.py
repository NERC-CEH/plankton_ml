"""Utilities for expressing our dataset as an intake catalog"""


def intake_yaml(
    test_url: str,
    catalog_url: str,
) -> str:
    """
    Write a minimal YAML template describing this as an intake datasource
    Example: plankton dataset made available through scivision, metadata
    https://raw.githubusercontent.com/alan-turing-institute/plankton-cefas-scivision/test_data_catalog/scivision.yml
    See the comments below for decisions about its structure
    """
    template = f"""
sources:
  test_image:
    description: Single test image from the plankton collection
    origin:
    driver: intake_xarray.image.ImageSource
    args:
      urlpath: ["{test_url}"]
      exif_tags: False
  plankton:
    description: A CSV index of all the images of plankton
    origin:
    driver: intake.source.csv.CSVSource
    args:
      urlpath: ["{catalog_url}"]
"""
    # coerce_shape: [256, 256]
    return template
