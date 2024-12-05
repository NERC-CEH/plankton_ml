import luigi
from cyto_ml.pipeline.pipeline_decollage import FlowCamPipeline


if __name__ == "__main__":
    luigi.build(
        [
            FlowCamPipeline(
                directory="./tests/fixtures/MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1",
                output_directory="./data/images_decollage",
                experiment_name="test_experiment",
                s3_bucket="test-upload-alba",
            )
        ],
        local_scheduler=False,
    )
