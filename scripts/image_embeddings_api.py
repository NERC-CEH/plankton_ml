"""Extract and store image embeddings from a collection in s3,
using an API that calls one or more off-the-shelf pre-trained models"""

import os
import logging
import yaml
from dotenv import load_dotenv
from cyto_ml.data.vectorstore import vector_store
import pandas as pd
import requests

logging.basicConfig(level=logging.info)
load_dotenv()

ENDPOINT = "http://localhost:8000/resnet18/"
PARAMS = os.path.join(os.path.abspath(os.path.dirname(__file__)), "params.yaml")

if __name__ == "__main__":

    # Limited to the Lancaster FlowCam dataset for now:
    image_bucket = yaml.safe_load(open(PARAMS))["collection"]
    catalog = f"{image_bucket}/catalog.csv"

    file_index = f"{os.environ.get('AWS_URL_ENDPOINT')}/{catalog}"
    df = pd.read_csv(file_index)

    # TODO - optional embedding length param at this point, it's not ideal
    collection = vector_store("sqlite", image_bucket, embedding_len=512)

    def store_embeddings(url):
        response = requests.post(ENDPOINT, data={"url": url}).json()
        if not "embeddings" in response:
            logging.error(response)
            raise

        response["url"] = url
        collection.add(**response)

    for _, row in df.iterrows():
        store_embeddings(row.item())
