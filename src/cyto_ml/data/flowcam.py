# Decollages flowcam images (breaks one large tiff into many small ones)
# Attempts to extract coordinate, date and depth information encoded in the filename
# Add add those properties to the resulting output in the EXIF headers
# where file path points to the flowcam data folder which has the collage .tifs and the .lst file inside
# Originally adapted from https://sarigiering.co/posts/extract-individual-particle-images-from-flowcam/
import argparse
import glob
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolExecuteError
from skimage.io import imread, imsave

logging.basicConfig(level=logging.INFO)


def lst_metadata(filename: str) -> pd.DataFrame:
    """
    Read the csv-ish ".lst" file from the FlowCam export
    Return a pandas dataframe
    """
    heads = pd.read_csv(filename, sep="|", nrows=53, skiprows=1)
    colNames = list(heads["num-fields"])
    meta = pd.read_csv(filename, sep="|", skiprows=55, header=None)
    meta.columns = colNames
    return meta


def window_slice(image: np.ndarray, x: int, y: int, height: int, width: int) -> np.ndarray:
    return image[y : y + height, x : x + width]  # noqa: E203


def headers_from_filename(filename: str) -> dict:
    """Attempt to extract lon/lat and date, option of depth, from filename
    Return a dict with key-value pairs for use as EXIF headers"""
    try:
        _, lat, lon, date, _ = parse_filename(filename)
    except ValueError as err:
        logging.debug(err)
        # No filename encoded metadata to return
        return {}
    return exif_headers(lat, lon, date)


def parse_filename(filename: str) -> tuple:
    """Attempt to extract file prefix, lon, lat, date, depth, from filename"""
    pattern = r"_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{8})(?:_(\d+))?"
    match = re.search(pattern, filename)
    if match:
        # We've left space for "depth" here
        # But all the observed values are not depths, they're like session IDs e.g. _1
        # TODO check this assumption with the folks in the lab
        lat, lon, date, depth = match.groups()

        # There could be an arbitrary number of underscores before the coords
        prefix = filename.split(lat)[0]
        return (prefix, lat, lon, date, depth)

    else:
        logging.warning(f"No coordinates or date found in filename: {filename}")
        return ()


def exif_headers(lon: float, lat: float, date: str, depth: Optional[int] = 0) -> dict:
    """
    Given lat, lon, date and option of depth, write and return a dict with EXIF standard tags as keys
    """
    headers = {}
    # https://exiftool.org/TagNames/GPS.html
    headers["GPSLatitude"] = lat
    headers["GPSLongitude"] = lon
    headers["DateTimeOriginal"] = date  # better to leave as date than pad with zero hours?
    if depth > 0:
        headers["GPSAltitude"] = depth  # can we use negative altitude as bathymetric depth?
    return headers


def write_headers(filename: str, headers: dict) -> bool:
    """
    Given a dictionary of EXIF tag keys and their values, write to filename
    Returns True if nothing has obviously gone wrong during this process
    """
    result = None
    try:
        with ExifToolHelper() as et:
            et.set_tags([filename], tags=headers, params=["-P", "-overwrite_original"])
        result = True
    # TODO try failures, observe them
    except ExifToolExecuteError as err:
        logging.warning(err)
        result = False
    return result


def read_headers(filename: str) -> dict:
    meta = {}
    with ExifToolHelper() as et:
        meta = et.get_metadata(filename)
    return meta


class FlowCamSession:
    """
    Bundle up all the logic of the decollage script so it can be run
    without passing commandline arguments
    """

    def __init__(self, directory: str, output_directory: str, experiment_name: str):
        """Implements the work of the decollage script:

        directory - path to a directory containing all the images for a FlowCam session
        output_directory - path to a directory to write the single images, create if needed
        experiment_name - a tag to use on the image files, could be superfluous
        """
        self.directory = directory
        self.output_directory = output_directory
        self.experiment_name = experiment_name

        self.read_metadata()
        self.output_dir()

        self.do_decollage()

    def read_metadata(self) -> None:
        self.metadata = {}

        files = glob.glob(f"{self.directory}/*.lst")

        if len(files) == 0:
            raise FileNotFoundError("no lst file in this directory")
        else:
            self.metadata = lst_metadata(files[0])

    def output_dir(self) -> None:
        # create a folder to save the output into
        if os.path.exists(self.output_directory):
            pass
        else:
            os.mkdir(self.output_directory)

    def do_decollage(self) -> None:
        """Not very lovely single function that replaces the work of the script.
        See cyto_ml.pipeline.pipeline_decollage - has the same code in it
        """
        # Reasonably assume that all images in a session have same spatio-temporal metadata
        # extract the coords, date, possibly depth from directory name
        collage_headers = headers_from_filename(self.directory)

        # decollage - rather than traverse the index and keep rereading large images,
        # filter by filename first and traverse that way, should speed up a lot
        for collage_file in self.metadata.collage_file.unique():
            collage = imread(f"{self.directory}/{collage_file}")

            df = self.metadata[self.metadata.collage_file == collage_file]

            for i in df.index:
                # extract vignette
                height = df["image_h"][i]
                width = df["image_w"][i]
                img_sub = window_slice(
                    collage,
                    df["image_x"][i],
                    df["image_y"][i],
                    height,
                    width,
                )
                # write EXIF metadata into the headers
                headers = collage_headers
                headers["ImageWidth"] = width
                headers["ImageHeight"] = height

                # save vignette to decollage folder
                # we probably need to write to the filesystem to then use exiftool
                output_file = f"{self.directory}/decollage/{self.experiment_name}_{i}.tif"
                imsave(output_file, img_sub)
                write_headers(output_file, headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FlowCam_DeCollager",
        description="Decollages flow cam images. requires pandas (pip install pandas) and cv2 (pip install opencv-python).",  # noqa: E501
    )
    parser.add_argument(
        "filePath",
        help="path to the flowcam data file which contains collage .tifs and an .lst file",
    )
    parser.add_argument("experimentName", help="name to append to each decollaged file")
    args = parser.parse_args()

    # Run the decollage process for a whole session
    FlowCamSession(args.filePath, f"{args.filePath}/decollage", args.experimentName)

    # TODO consider squirting the output straight into the object store API

    # TODO decide whether to do anything with the analytic metadata (circularity etc)
    # We could pop it into a sqlite store at this stage, but want the file linkages
