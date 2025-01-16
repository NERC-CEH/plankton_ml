import pickle
from typing import Any, Dict, List, Literal, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from resnet50_cefas import load_model

from cyto_ml.data.image import load_image_from_url
from cyto_ml.models.utils import flat_embeddings

# Label Studio ML limits our ability to manage sessions -
# see cyto_ml/models/api.py for a FastAPI version that's more considered
resnet50_model = load_model(strip_final_layer=True)


class ImageNotFoundError(Exception):
    pass


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model"""

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
        print(f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}""")

        # TODO as below, check what the response format should really be (try it!)
        predictions = []
        for task in tasks:
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

        features = resnet50_model(load_image_from_url(image_url))
        embeddings = flat_embeddings(features)
        # Classify embeddings (KNN to start, many improvements possible!) and return a label
        label = self.embeddings_predict(embeddings)
        # TODO check what the return format should be - does ModelResponse handle this?
        return label

    def embeddings_predict(self, embeddings: List[List[float]]) -> List[str]:
        """Predict labels from embeddings
        See cyto_ml/visualisation/pages/02_kmeans.py for usage for a collection
        See scripts/cluster.py for the model build and save
        """
        # TODO load this from config, add to Dockerfile
        fitted = pickle.load(open("../models/kmeans-untagged-images-lana.pkl", "rb"))
        label = fitted.predict(embeddings)
        return label

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
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f"New data: {self.get('my_data')}")
        print(f"New model version: {self.get('model_version')}")

        print("fit() completed successfully.")
