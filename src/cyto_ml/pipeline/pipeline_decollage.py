import glob
import logging
import os
import re
from datetime import datetime

import boto3
import luigi
import pandas as pd
import requests
from dotenv import load_dotenv
from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolExecuteError
from skimage.io import imread, imsave

# Set up logging
logging.basicConfig(level=logging.INFO)


# Load AWS credentials and S3 bucket name from .env file
# Rather than depend on the presence of credentials.json in the package
load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_URL_ENDPOINT = os.environ.get("AWS_URL_ENDPOINT", "")

# S3 Client
s3 = boto3.client("s3",
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                  endpoint_url=AWS_URL_ENDPOINT)


# Utility functions (kept as is from your original code)
def lst_metadata(filename: str) -> pd.DataFrame:
    heads = pd.read_csv(filename, sep="|", nrows=53, skiprows=1)
    colNames = list(heads["num-fields"])
    meta = pd.read_csv(filename, sep="|", skiprows=55, header=None)
    meta.columns = colNames
    return meta


def window_slice(image, x, y, height, width):
    return image[y:y + height, x:x + width]


def headers_from_filename(filename: str) -> dict:
    headers = {}
    pattern = r"_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{8})(?:_(\d+))?"
    match = re.search(pattern, filename)
    if match:
        lat, lon, date, depth = match.groups()
        headers["GPSLatitude"] = lat
        headers["GPSLongitude"] = lon
        headers["DateTimeOriginal"] = date
        headers["GPSAltitude"] = depth
    return headers


def write_headers(filename: str, headers: dict) -> bool:
    result = None
    try:
        with ExifToolHelper() as et:
            et.set_tags([filename], tags=headers, params=["-P", "-overwrite_original"])
        result = True
    except ExifToolExecuteError as err:
        logging.warning(err)
        result = False
    return result


class ReadMetadata(luigi.Task):
    """
    Task to read metadata from the .lst file.
    """
    directory = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'{self.directory}/metadata.csv')

    def run(self):
        files = glob.glob(f"{self.directory}/*.lst")
        if len(files) == 0:
            raise FileNotFoundError("No .lst file found in this directory.")

        metadata = lst_metadata(files[0])
        metadata.to_csv(self.output().path, index=False)
        logging.info(f"Metadata read and saved to {self.output().path}")


class CreateOutputDirectory(luigi.Task):
    """
    Task to create the output directory if it does not exist.
    """
    output_directory = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_directory)

    def run(self):
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
            logging.info(f"Output directory created: {self.output_directory}")
        else:
            logging.info(f"Output directory already exists: {self.output_directory}")


class DecollageImages(luigi.Task):
    """
    Task that processes the large TIFF image, extracts vignettes, and saves them with EXIF metadata.
    """
    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()

    def requires(self):
        return [ReadMetadata(self.directory), CreateOutputDirectory(self.output_directory)]

    def output(self):
        date = datetime.today().date()
        return luigi.LocalTarget(f'./decollage_complete_{date}.txt')

    def run(self):
        metadata = pd.read_csv(self.input()[0].path)
        collage_headers = headers_from_filename(self.directory)

        # Loop through unique collage files and slice images
        for collage_file in metadata.collage_file.unique():
            collage = imread(f"{self.directory}/{collage_file}")
            df = metadata[metadata.collage_file == collage_file]

            for i in df.index:
                height = df["image_h"][i]
                width = df["image_w"][i]
                img_sub = window_slice(collage, df["image_x"][i], df["image_y"][i], height, width)

                # Add EXIF metadata
                headers = collage_headers
                headers["ImageWidth"] = width
                headers["ImageHeight"] = height

                # Save vignette with EXIF metadata
                output_file = f"{self.output_directory}/{self.experiment_name}_{i}.tif"
                imsave(output_file, img_sub)
                write_headers(output_file, headers)

        with self.output().open('w') as f:
            f.write('Decollage complete')


class UploadDecollagedImagesToS3(luigi.Task):
    """
    Task to upload decollaged images to an S3 bucket.
    """
    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    s3_bucket = luigi.Parameter()

    def requires(self):
        return DecollageImages(
            directory=self.directory,
            output_directory=self.output_directory,
            experiment_name="test_experiment"
        )

    def output(self):
        date = datetime.today().date()
        return luigi.LocalTarget(f'./s3_upload_complete_{date}.txt')

    def run(self):
        # Collect the list of decollaged image files from the output of DecollageImages
        image_files = glob.glob(f"{self.output_directory}/*.tif")

        # Prepare the files for uploading
        files = [("files", (open(image_file, 'rb'))) for image_file in image_files]

        # Prepare the payload for the API request
        payload = {
            "bucket_name": self.s3_bucket,
        }

        # API endpoint
        url = "http://localhost:8080/upload/"

        logging.info(f"Sending {len(image_files)} files to {url}")

        try:
            # Send the POST request to the API
            response = requests.post(url, files=files, data=payload)

            # Check if the request was successful
            if response.status_code == 200:
                logging.info("Files successfully uploaded via API.")
                with self.output().open('w') as f:
                    f.write("API upload complete")
            else:
                logging.error(f"API upload failed with status code {response.status_code}")
                logging.error(response.content)
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to upload files to API: {e}")
            raise e

        with self.output().open('w') as f:
            f.write("S3 upload complete")


class FlowCamPipeline(luigi.WrapperTask):
    """
    Main wrapper task to execute the entire pipeline.
    """
    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()
    s3_bucket = luigi.Parameter()

    def requires(self):
        return UploadDecollagedImagesToS3(
            directory=self.directory,
            output_directory=self.output_directory,
            s3_bucket=self.s3_bucket
        )


# To run the pipeline
if __name__ == '__main__':
    luigi.run([
        "FlowCamPipeline",
        # "--local-scheduler",
        "--directory", "/home/albseg/scratch/plankton_pipeline_luigi/data/19_10_Tank25_blanksremoved",
        "--output-directory", "/home/albseg/scratch/plankton_pipeline_luigi/data/images_decollage",
        "--experiment-name", "test",
        "--s3-bucket", "test-upload-alba"
    ])
