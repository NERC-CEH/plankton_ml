import luigi
from pipeline.pipeline_decollage import FlowCamPipeline


if __name__ == '__main__':
    luigi.build([
        FlowCamPipeline(
            directory="./data/19_10_Tank25_blanksremoved",
            output_directory="./data/images_decollage",
            experiment_name="test_experiment",
            s3_bucket="test-upload-alba"
        )
    ], local_scheduler=False)
