import argparse
import logging
from pathlib import Path
import datetime
import os

from cryojax.data import RelionParticleParameterFile, RelionParticleStackDataset

from src.whitening import whiten_relion_dataset

logger = logging.getLogger()
logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
logger_fname = logger_fname + ".log"
fhandler = logging.FileHandler(filename=logger_fname, mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)


def main(
    path_to_relion_project: Path,
    path_to_starfile: Path,
    images_per_mrc: int,
    output_folder: Path,
    overwrite: bool,
):
    relion_dataset = RelionParticleStackDataset(
        RelionParticleParameterFile(path_to_starfile, mode="r"),
        path_to_relion_project=path_to_relion_project,
        mode="r",
    )

    whiten_relion_dataset(
        path_to_new_starfile=os.path.join(output_folder, "whitened_particles.star"),
        path_to_new_relion_project=output_folder,
        relion_dataset=relion_dataset,
        images_per_mrc=images_per_mrc,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    # Setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_starfile",
        type=str,
        help="Path to the STAR file particle stack of segmented MTs.",
    )
    parser.add_argument(
        "path_to_relion_project",
        type=str,
        help="Path to the RELION project directory.",
    )

    parser.add_argument(
        "--images-per-mrc",
        type=int,
        help="Number of images per MRC file.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files.",
        default=False,
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, help="Output folder to write results."
    )
    parser.add_argument("-l", "--log", type=str, help="Set level of logger.")

    # Parse arguments
    args = parser.parse_args()
    # Unpack parser
    path_to_starfile = Path(args.path_to_starfile)
    path_to_relion_project = Path(args.path_to_relion_project)
    images_per_mrc = args.images_per_mrc
    overwrite = args.overwrite

    log_level = args.log or "INFO"

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    # Run
    main(
        path_to_relion_project,
        path_to_starfile,
        images_per_mrc,
        output_folder,
        overwrite,
    )
