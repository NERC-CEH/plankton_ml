import pandas as pd
from skimage.io import imread
from cyto_ml.data.flowcam import (
    lst_metadata,
    window_slice,
    headers_from_filename,
    write_headers,
    read_headers,
)


def test_lst_metadata(lst_file):
    df = lst_metadata(lst_file)
    assert isinstance(df, pd.DataFrame)


def test_window_slice(collage_file):
    img = imread(collage_file)
    win = window_slice(img, 5, 5, 25, 50)
    assert win.shape == (25, 50, 3)


def test_headers_from_filename(collage_file):
    h = headers_from_filename(collage_file)
    assert "GPSLatitude" in h and h["GPSLatitude"]


def test_write_headers(exiftest_file):
    # Check we don't have a tagged version from a previous run

    tags = read_headers(exiftest_file)
    assert "EXIF:GPSLatitude" not in tags

    # Note this has to be a "valid" EXIF header
    write_headers(exiftest_file, {"GPSLatitude": "42"})

    meta = read_headers(exiftest_file)

    assert "EXIF:GPSLatitude" in meta[0]

    meta = write_headers("nonexistent_file.tif", {"GPSLatitude": "42"})
