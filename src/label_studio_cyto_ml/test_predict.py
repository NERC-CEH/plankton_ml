import json
from typing import Any  # sorry, this is all for servicing ruff pickiness about type annotation :/

import pytest
from label_studio_ml.api import init_app
from model import NewModel

sample_body = '{"tasks": [{"id": 2, "data": {"image": "s3://untagged-images-lana-ls/MicrobialMethane_MESO_Tank7_54.0143_-2.7770_18052023_1_10.png"}, "meta": {}, "created_at": "2025-02-05T10:47:24.857459Z", "updated_at": "2025-02-05T10:47:24.857473Z", "is_labeled": false, "overlap": 1, "inner_id": 2, "total_annotations": 0, "cancelled_annotations": 0, "total_predictions": 0, "comment_count": 0, "unresolved_comment_count": 0, "last_comment_updated_at": null, "project": 1, "updated_by": null, "file_upload": null, "comment_authors": [], "annotations": [], "predictions": []}], "project": "1.1738687422", "label_config": "<View>\\n  <Image name=\\"image\\" value=\\"$image\\" zoom=\\"true\\" zoomControl=\\"true\\" rotateControl=\\"true\\"/>\\n  <Choices name=\\"organism_type\\" toName=\\"image\\" label=\\"single\\" required=\\"false\\" showInline=\\"true\\">\\n    <Choice value=\\"Not-plankton\\"/>\\n    <Choice value=\\"Plankton\\"/>\\n    <Choice value=\\"Debris\\"/>\\n  </Choices>\\n  <Choices name=\\"morphology\\" toName=\\"image\\" label=\\"multiple\\" required=\\"false\\" showInline=\\"true\\">\\n    <Choice value=\\"Mucilage\\"/>\\n    <Choice value=\\"Flagella\\"/>\\n    <Choice value=\\"Cilia\\"/>\\n    <Choice value=\\"Aerotopes\\"/>\\n    <Choice value=\\"Akinetes\\"/>\\n    <Choice value=\\"Heterocytes\\"/>\\n    <Choice value=\\"Theca/test/exoskeletal structures\\"/>\\n    <Choice value=\\"Eggs\\"/>\\n    <Choice value=\\"Ephippia\\"/>\\n  </Choices>\\n  <Choices name=\\"life_form\\" toName=\\"image\\" label=\\"single\\" required=\\"false\\" showInline=\\"true\\">\\n    <Choice value=\\"Unicellular\\"/>\\n    <Choice value=\\"Colony\\"/>\\n    <Choice value=\\"Filament\\"/>\\n  </Choices>\\n  <Choices name=\\"shape\\" toName=\\"image\\" label=\\"multiple\\" required=\\"false\\" showInline=\\"true\\">\\n    <Choice value=\\"Spiky\\"/>\\n    <Choice value=\\"Round\\"/>\\n    <Choice value=\\"Rod-like\\"/>\\n  </Choices>\\n  <!--<Taxonomy name=\\"taxonomy\\" toName=\\"image\\" apiUrl=\\"s3://taxonomy-ls/taxonomy.json\\" placeholder=\'Choose taxonomy\'></Taxonomy>-->\\n  <TextArea name=\\"ta\\" toName=\'image\' placeholder=\\"Write custom taxonomic classification here.\\"></TextArea>\\n</View>", "params": {"login": null, "password": null, "context": null}}'  #noqa: E501

minimal_body = """{"tasks": [{"id":1, "data": {"image": "s3://untagged-images-lana-ls/MicrobialMethane_MESO_Tank7_54.0143_-2.7770_18052023_1_10.png"}},
                          {"id":2, "data": {"image": "s3://untagged-images-lana-ls/MicrobialMethane_MESO_Tank7_54.0143_-2.7770_18052023_1_10.png"}}],
                           "label_config":"<Choice value=\\"Debris\\"\\/>"}"""
@pytest.fixture
def web_app() -> Any:
    app = init_app(model_class=NewModel)
    return app

@pytest.fixture()
def client(web_app: Any) -> Any:
    return web_app.test_client()

def test_predict(client: Any) -> None:
    response = client.post("/predict", json=json.loads(sample_body))
    assert response.status_code == 200
    assert len(response.json["results"]) == 1

    response = client.post("/predict", json=json.loads(minimal_body))
    assert response.status_code == 200

    assert len(response.json["results"]) == 2
