"""Try to use the scivision pretrained model and tools against this collection"""

import os
import logging
import yaml
from dotenv import load_dotenv
from cyto_ml.models.utils import flat_embeddings
from cyto_ml.data.image import load_image_from_url

from resnet50_cefas import load_model
from cyto_ml.data.vectorstore import vector_store
import pandas as pd

logging.basicConfig(level=logging.info)
load_dotenv()


if __name__ == "__main__":

    # Limited to the Lancaster FlowCam dataset for now:
    image_bucket = yaml.safe_load(open("params.yaml"))["collection"]
    catalog = f"{image_bucket}/catalog.csv"

    file_index = f"{os.environ.get('AWS_URL_ENDPOINT')}/{catalog}"
    df = pd.read_csv(file_index)

    collection = vector_store(image_bucket)

    model = load_model(strip_final_layer=True)

    def store_embeddings(row):
        try:
            image_data = load_image_from_url(row.Filename)
        except ValueError as err:
            # TODO diagnose and fix for this happening, in rare circumstances:
            # (would be nice to know rather than just buffer the image and add code)
            # File "python3.9/site-packages/PIL/PcdImagePlugin.py", line 34, in _open
            #   self.fp.seek(2048)
            # File "python3.9/site-packages/fsspec/implementations/http.py", line 745, in seek
            # raise ValueError("Cannot seek streaming HTTP file")
            # Is this still reproducible? - JW
            logging.info(err)
            logging.info(row.Filename)
            return
        except OSError as err:
            logging.info(err)
            logging.info(row.Filename)
            return

        embeddings = flat_embeddings(model(image_data))

        collection.add(
            documents=[row.Filename],
            embeddings=[embeddings],
            ids=[row.Filename],  # must be unique
            # Note - optional arg name is "metadatas" (we don't have any)
        )

    for _, row in df.iterrows():
        store_embeddings(row)
