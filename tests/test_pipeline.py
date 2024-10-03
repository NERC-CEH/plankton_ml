import os
import pytest
import pandas as pd
import numpy as np
from skimage.io import imsave
import luigi
from cyto_ml.pipeline.pipeline_decollage import ReadMetadata, DecollageImages, UploadDecollagedImagesToS3


@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory using pytest's tmp_path fixture."""
    return tmp_path


def test_read_metadata(temp_dir):
    # Create a mock .lst file for testing
    lst_file_content = """num-fields|value
    field1|val1
    field2|val2
    """
    lst_file_path = os.path.join(temp_dir, 'test.lst')
    with open(lst_file_path, 'w') as f:
        f.write(lst_file_content)

    # Run the ReadMetadata task
    task = ReadMetadata(directory=str(temp_dir))
    luigi.build([task], local_scheduler=True)

    # Check if metadata.csv was created
    output_file = task.output().path
    assert os.path.exists(output_file), "Metadata CSV file should be created."
    df = pd.read_csv(output_file)
    print(df)
    assert len(df) == 2, "The metadata CSV should have two fields."


def test_decollage_images(temp_dir):
    # Create mock metadata
    metadata = pd.DataFrame({
        "collage_file": ["test_collage.tif"],
        "image_x": [0],
        "image_y": [0],
        "image_h": [100],
        "image_w": [100]
    })
    metadata.to_csv(os.path.join(temp_dir, "metadata.csv"), index=False)

    # Create a mock TIFF image
    img_path = os.path.join(temp_dir, "test_collage.tif")
    img = np.zeros((200, 200), dtype=np.uint8)
    imsave(img_path, img)

    # Run the DecollageImages task
    task = DecollageImages(directory=str(temp_dir), output_directory=str(temp_dir), experiment_name="test_experiment")
    luigi.build([task], local_scheduler=True)

    # Check if the output image was created
    output_image = os.path.join(temp_dir, "test_experiment_0.tif")
    assert os.path.exists(output_image), "Decollaged image should be created."


def test_upload_to_api(temp_dir, mocker):
    # Mock the DecollageImages output using pytest-mock
    mock_output = mocker.patch('pipeline.pipeline_decollage.DecollageImages.output')
    mock_output.return_value = [os.path.join(temp_dir, "test_experiment_0.tif")]

    # Mock the requests.post to simulate the API response
    mock_post = mocker.patch('pipeline.pipeline_decollage.requests.post')
    mock_post.return_value.status_code = 200

    task = UploadDecollagedImagesToS3(
        directory=str(temp_dir),
        output_directory=str(temp_dir),
        s3_bucket="mock_bucket",
        s3_folder="mock_folder"
    )

    luigi.build([task], local_scheduler=True)

    # Check if the task's output file was created (indicating success)
    assert os.path.exists(task.output().path), "S3 upload completion file should be created."
    mock_post.assert_called_once()  # Ensure the API was called
