"""Extract and store image embeddings from a collection in s3,
using an off-the-shelf pre-trained model"""

import os
import logging
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
from cyto_ml.models.utils import flat_embeddings, resnet18
from cyto_ml.data.image import load_image_from_url

from cyto_ml.data.vectorstore import vector_store
import pandas as pd

logging.basicConfig(level=logging.info)
load_dotenv()

STATE_FILE='../models/ResNet_18_3classes_RGB.pth'

if __name__ == "__main__":

    # Limited to the Lancaster FlowCam dataset for now:
    image_bucket = yaml.safe_load(open("params.yaml"))["collection"]
    file_index = f"{image_bucket}.csv"

    # We have a static file index, written by image_index.py
    df = pd.read_csv(file_index)

    # Keep a sqlite db per-collection. Plan to sync these to s3 with DVC
    db_dir = '../data'
    if not os.path.exists(db_dir):
        os.mkdir(db_dir)
    collection = vector_store("sqlite", f"{db_dir}/{image_bucket}.db")

    # Turing Inst 3-class lightweight model needs downloaded manually.
    # Please see https://github.com/alan-turing-institute/ViT-LASNet/issues/2
    model = resnet18(num_classes=3, filename=STATE_FILE, strip_final_layer=True)

    def store_embeddings(url):
        try:
            image_data = load_image_from_url(url)
        except ValueError as err:
            # TODO diagnose and fix for this happening, in rare circumstances:
            # (would be nice to know rather than just buffer the image and add code)
            # File "python3.9/site-packages/PIL/PcdImagePlugin.py", line 34, in _open
            #   self.fp.seek(2048)
            # File "python3.9/site-packages/fsspec/implementations/http.py", line 745, in seek
            # raise ValueError("Cannot seek streaming HTTP file")
            # Is this still reproducible? - JW
            logging.info(err)
            logging.info(url)
            return
        except OSError as err:
            logging.info(err)
            logging.info(row.Filename)
            return

        embeddings = flat_embeddings(model(image_data))

        collection.add(
            url=url,
            embeddings=embeddings,
        )

    for _, row in tqdm(df.iterrows()):
        image_url = f"{os.environ['AWS_URL_ENDPOINT']}/{image_bucket}/{row[0]}"
        store_embeddings(image_url)
