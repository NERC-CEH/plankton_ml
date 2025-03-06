import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse, PredictionValue
from resnet50_cefas import load_model

from cyto_ml.data.image import load_image_from_url
from cyto_ml.models.utils import flat_embeddings

# Set AWS_URL_ENDPOINT in here
# Used to convert s3:// URLs coming from Label Studio to https:// URLs
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
# Label Studio ML limits our ability to manage sessions -
# see cyto_ml/models/api.py for a FastAPI version that's more considered
resnet50_model = load_model(strip_final_layer=True)
default_classifier = "untagged-images-lana"


class ImageNotFoundError(Exception):
    pass


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model"""

    # Assumes models are stored in './models' in root of project
    model_path = Path(__file__).parent.parent.parent / "models"

    def setup(self) -> None:
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logging.info(f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}""")

        # TODO as below, check what the response format should really be (try it!)
        predictions = []
        for task in tasks:
            logging.debug(task)
            try:
                annotated = self.predict_task(task)
                predictions.append(annotated)
            except KeyError as err:
                # Return 500 with detail
                raise (err)
            except ImageNotFoundError as err:
                # Return 404 with detail
                raise (err)

        # https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/response.py

        return ModelResponse(predictions=predictions)

    def convert_url(self, url: str) -> str:
        """Convert an s3:// URL to an https:// URL, if a https AWS_URL_ENDPOINT is set in .env
        If it's not set, then return the URL unchanged (useful for testing)
        """
        if "https" not in os.getenv("AWS_URL_ENDPOINT"):
            return url
        if url.startswith("s3://"):
            return url.replace("s3://", f"{os.getenv('AWS_URL_ENDPOINT')}/")
        return url

    def bucket_from_url(self, url: str) -> str:
        """Extract the bucket from an s3:// URL.
        If it's not inferrable or this isn't an s3 URL, then return a default model (for testing)
        """
        if not url.startswith("s3://"):
            return default_classifier
        try:
            bucket = url.split("/")[2]
        except IndexError:
            raise ImageNotFoundError(f"Could not find bucket in {url}")
        if bucket and "-ls" in bucket:
            bucket = bucket.replace("-ls", "")

        return bucket

    def predict_task(self, task: dict) -> dict:
        """Receive a single task definition as described here https://labelstud.io/guide/task_format.html
        Return the task decorated with predictions as described here
        https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks
        """
        # We use two models here:
        # Extract image embeddings with a ResNet
        try:
            image_url = task.get("data").get("image")
        except KeyError as err:
            raise (err)

        features = resnet50_model(load_image_from_url(self.convert_url(image_url)))
        embeddings = flat_embeddings(features)

        # Classify embeddings (KNN to start, many improvements possible!) and return a label
        # This allows us one prediction model per bucket, but it could be an ensemble
        bucket_name = self.bucket_from_url(image_url)

        label = self.embeddings_predict(embeddings, model=bucket_name)
        # Debugging purposes - label everything as Debris
        return PredictionValue(
            result=[
                {
                    "id": task["id"],
                    "value": "Debris",
                    "from_name": "organism_type",
                    "type": "Choices",
                    "to_name": "image",
                }
            ]
        )

    def embeddings_predict(self, embeddings: List[List[float]], model: Optional[str] = "") -> List[str]:
        """Predict labels from embeddings
        See cyto_ml/visualisation/pages/02_kmeans.py for usage for a collection
        See scripts/cluster.py for the model build and save.
        Args:
            embeddings: List of embeddings
            model: The name of the model to use (based on bucket name)
        """

        # "naming convention" is {model type}-{bucket name}
        fitted = pickle.load(open(f"{self.model_path}/kmeans-{model}.pkl", "rb"))
        label = fitted.predict([embeddings])[0]

        # The prediction format should be this, a dict
        # model_version: Optional[Any] = None
        # score: Optional[float] = 0.00
        # result: Optional[List[Union[Dict[str, Any], Region]]]
        # https://labelstud.io/guide/predictions#Add-results-to-the-predictions-array
        # TODO map model labels to UI labels - there is an API for this if we want to do it right
        return int(label)

    def fit(
        self,
        event: Literal["ANNOTATION_CREATED", "ANNOTATION_UPDATED", "START_TRAINING"],
        data: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        """

        pass