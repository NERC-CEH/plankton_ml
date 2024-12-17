import argparse
import luigi
from cyto_ml.pipeline.pipeline_decollage import FlowCamPipeline
from cyto_ml.data.flowcam import parse_filename
import logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog='Luigi FlowCam pipeline',
    description='Triggers the process of decollaging images, embedding spatio-temporal metadata in headers, and uploading to s3 storage',
    )

    parser.add_argument('-o', '--output_directory', type=str, default="./data/images_decollage", help="directory for intermediate data")
    parser.add_argument('-d', '--directory', type=str, default="./tests/fixtures/MicrobialMethane_MESO_Tank10_54.0143_-2.7770_04052023_1", help='Enable verbose mode')
    parser.add_argument('-s', '--s3_bucket', type=str, default="untagged-images-lana")
    parser.add_argument('-e', '--experiment_name', type=str)
    args = parser.parse_args()

    experiment = args.experiment_name
    if not experiment:
        try:
            prefix, _, _, date, _ = parse_filename(args.directory)
            experiment = ''.join([prefix, date])
        except ValueError as err:
            logging.info("Could't figure out experiment name and date from {args.directory} - please call this with --experiment_name to set this")
            logging.debug(err)
            exit

    luigi.build(
        [
            FlowCamPipeline(
                directory=args.directory,
                output_directory=args.output_directory,
                experiment_name=experiment,
                s3_bucket=args.s3_bucket,
            )
        ],
        local_scheduler=False,
    )
