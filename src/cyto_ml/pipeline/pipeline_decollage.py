import glob
import logging
import os
from datetime import datetime
from typing import List

import luigi
import pandas as pd
import requests
from dotenv import load_dotenv
from skimage.io import imread, imsave

from cyto_ml.data.decollage import headers_from_filename, lst_metadata, window_slice, write_headers

# Set up logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

# API endpoint
API_URL = "http://localhost:8000/upload/"


class ReadMetadata(luigi.Task):
    """
    Task to read metadata from the .lst file.
    """

    directory = luigi.Parameter()

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(f"{self.directory}/metadata.csv")

    def run(self) -> None:
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

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(self.output_directory)

    def run(self) -> None:
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
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

    def requires(self) -> List[luigi.Task]:
        return [ReadMetadata(self.directory), CreateOutputDirectory(self.output_directory)]

    def output(self) -> luigi.Target:
        date = datetime.today().date()
        return luigi.LocalTarget(f"{self.directory}/decollage_complete_{date}.txt")

    def run(self) -> None:
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

        with self.output().open("w") as f:
            f.write("Decollage complete")


class UploadDecollagedImagesToS3(luigi.Task):
    """
    Task to upload decollaged images to an S3 bucket.
    """

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    s3_bucket = luigi.Parameter()
    api_url = luigi.Parameter(default=API_URL)

    def requires(self) -> List[luigi.Task]:
        return DecollageImages(
            directory=self.directory, output_directory=self.output_directory, experiment_name="test_experiment"
        )

    def output(self) -> luigi.Target:
        date = datetime.today().date()
        return luigi.LocalTarget(f"{self.directory}/s3_upload_complete_{date}.txt")

    def run(self) -> None:
        # Collect the list of decollaged image files from the output of DecollageImages
        image_files = glob.glob(f"{self.output_directory}/*.tif")

        # Prepare the files for uploading
        files = [("files", (open(image_file, "rb"))) for image_file in image_files]

        # Prepare the payload for the API request
        payload = {
            "bucket_name": self.s3_bucket,
        }

        logging.info(f"Sending {len(image_files)} files to {self.api_url}")

        try:
            # Send the POST request to the API
            response = requests.post(self.api_url, files=files, data=payload)

            # Check if the request was successful
            if response.status_code == 200:
                logging.info("Files successfully uploaded via API.")
                with self.output().open("w") as f:
                    f.write("API upload complete")
            else:
                logging.error(f"API upload failed with status code {response.status_code}")
                logging.error(response.content)
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to upload files to API: {e}")
            raise e

        with self.output().open("w") as f:
            f.write("S3 upload complete")


class FlowCamPipeline(luigi.WrapperTask):
    """
    Main wrapper task to execute the entire pipeline.
    """

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()
    s3_bucket = luigi.Parameter()
    api_url = luigi.Parameter(default=API_URL)

    def requires(self) -> luigi.Task:
        return UploadDecollagedImagesToS3(
            directory=self.directory,
            output_directory=self.output_directory,
            s3_bucket=self.s3_bucket,
            api_url=self.api_url,
        )


# To run the pipeline
if __name__ == "__main__":
    luigi.run(
        [
            "FlowCamPipeline",
            # "--local-scheduler",
            "--directory",
            "./data/19_10_Tank25_blanksremoved",
            "--output-directory",
            "./data/images_decollage",
            "--experiment-name",
            "test",
            "--s3-bucket",
            "test-upload-alba",
        ]
    )
