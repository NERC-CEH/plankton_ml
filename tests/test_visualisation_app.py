"""
Unit tests for the visualisation application.
"""

from unittest import TestCase, mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from cyto_ml.visualisation.app import create_figure


class TestClusteringApp(TestCase):
    """
    Test class for the visualisation streamlit app.
    """

    def setUp(self):
        """
        Create some dummy data for testing.
        """
        self.data = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [10, 11],
                "topic_number": [1, 2],
                "doc_id": ["id1", "id2"],
                "short_title": ["stitle1", "stitle2"],
            }
        )

    def test_app_starts(self):
        """
        Test the streamlit app starts.

        Note: current support for streamlit testing doesn;t currently allow to
        mimic user interactions with the visualisation.
        """
        with mock.patch(
            "cyto_ml.visualisation.app.image_ids",
            return_value=self.data,
        ):
            AppTest.from_file("src/cyto_ml/visualisation/app.py").run(
                timeout=30
            )

    def test_create_figure(self):
        """
        Ensure figure is created appropriately using dummy data.

        """
        fig = create_figure(self.data)

        scatter_data = fig.data[0]
        assert scatter_data.type == "scatter", "The plot type should be scatter"
        assert all(scatter_data.x == self.data["x"]), "X data should match"
        assert all(scatter_data.y == self.data["y"]), "Y data should match"
