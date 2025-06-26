import logging
from typing import Any, Tuple, Dict, Optional

import jax
import equinox as eqx
from cryojax.data import RelionParticleParameterFile, RelionParticleStackDataset
from cryojax.image import downsample_with_fourier_cropping
from jaxtyping import Int, Array, Float

from .utils import create_dataloader, get_images_per_mrc


def downsample_relion_dataset(
    path_to_new_starfile: str,
    path_to_new_relion_project: str,
    relion_dataset: RelionParticleStackDataset,
    downsampling_factor: float | int,
    images_per_mrc: Optional[int] = None,
    *,
    overwrite: bool = True,
    mrcfile_settings: Dict[str, Any] = None,
    batch_size: int = 1,
) -> RelionParticleStackDataset:
    if mrcfile_settings is None:
        mrcfile_settings = {
            "prefix": "downsampled",
            "delimiter": "_",
            "overwrite": overwrite,
        }

    logging.info(
        f"Downsampling RELION dataset with downsampling factor {downsampling_factor}."
    )
    if images_per_mrc is None:
        images_per_mrc = get_images_per_mrc(relion_dataset.parameter_file)

    logging.info(f"Writing {images_per_mrc} images per MRC file.")

    logging.debug("Validating inputs...")
    _validate_inputs(relion_dataset.parameter_file, downsampling_factor)
    logging.debug("Inputs validated.")

    logging.info("Updating optical parameters in parameter file...")
    new_parameter_file = _update_parameter_file(
        relion_dataset.parameter_file, downsampling_factor
    )
    logging.info("Optical parameters updated.")

    new_relion_dataset = RelionParticleStackDataset(
        RelionParticleParameterFile(
            path_to_starfile=path_to_new_starfile,
            mode="w",
            exists_ok=overwrite,
        ),
        path_to_relion_project=path_to_new_relion_project,
        mode="w",
        mrcfile_settings=mrcfile_settings,
    )

    dataloader = create_dataloader(relion_dataset, batch_size=images_per_mrc)
    logging.info("Starting downsampling of images...")
    for batch in dataloader:
        downsampled_images = _downsample_images_with_fourier_cropping(
            batch["stack"]["images"], downsampling_factor, batch_size
        )
        new_relion_dataset.append(
            {
                "images": downsampled_images,
                "parameters": new_parameter_file[batch["index"]],
            }
        )
    logging.info("Downsampling completed. Saving new dataset...")

    new_relion_dataset.parameter_file.save(overwrite=True)
    logging.info(
        f"New dataset saved to {path_to_new_starfile} and {path_to_new_relion_project}."
    )
    return new_relion_dataset


@eqx.filter_jit
def _downsample_images_with_fourier_cropping(
    images: Float[Array, "n_images y x"],
    downsampling_factor: Tuple[Int, Int],
    batch_size: int,
    outputs_real_space: bool = True,
) -> Float[Array, "n_images y_ds x_ds"]:
    return jax.lax.map(
        lambda x: downsample_with_fourier_cropping(
            x, downsampling_factor, outputs_real_space=outputs_real_space
        ),
        images,
        batch_size=batch_size,
    )


def _update_parameter_file(
    parameter_file: RelionParticleParameterFile, downsampling_factor: float | int
):
    parameter_file = parameter_file.copy()

    # compute new pixel size
    instrument_config = parameter_file[0]["instrument_config"]

    new_pixel_size = instrument_config.pixel_size * downsampling_factor
    new_box_size = int(instrument_config.shape[0] / downsampling_factor)
    logging.info(f"New pixel size: {new_pixel_size}, New box size: {new_box_size}.")

    # update starfile data
    starfile_data = parameter_file.copy().starfile_data
    starfile_data["optics"]["rlnImagePixelSize"] = new_pixel_size
    starfile_data["optics"]["rlnImageSize"] = new_box_size
    starfile_data["particles"].drop("rlnImageName", axis=1, inplace=True)
    parameter_file.starfile_data = starfile_data

    return parameter_file


def _validate_inputs(
    parameter_file: RelionParticleParameterFile, downsampling_factor: float | int
):
    instrument_config = parameter_file[0]["instrument_config"]
    shape = instrument_config.shape
    if not (shape[0] == shape[1]):
        raise ValueError("Images must be square.")
    if downsampling_factor < 1.0:
        raise ValueError("`downsampling_factor` must be greater than 1.0")
    return
