"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import json
from typing import Generator

import pytest
from flask.testing import FlaskClient

from label_studio_cyto_ml.model import NewModel


@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    from _wsgi import init_app

    app = init_app(model_class=NewModel)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.skip(reason="Skipping until we define a model")
def test_predict(client: FlaskClient) -> None:
    request = {
        "tasks": [
            {
                "data": {
                    # Your input test data here
                }
            }
        ],
        # Your labeling configuration here
        "label_config": "<View></View>",
    }

    expected_response = {
        "results": [
            {
                # Your expected result here
            }
        ]
    }

    response = client.post("/predict", data=json.dumps(request), content_type="application/json")
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response == expected_response
