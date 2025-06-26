from typing import Any, Dict, Optional
import os

import jax.numpy as jnp
from cryojax.data import RelionParticleParameterFile, RelionParticleStackDataset
import cryojax.image.operators as op
from cryojax.image import rfftn, irfftn
import jax_dataloader as jdl

from utils import create_dataloader, get_images_per_mrc


def whiten_relion_dataset(
    path_to_new_starfile: str,
    path_to_new_relion_project: str,
    relion_dataset: RelionParticleStackDataset,
    images_per_mrc: Optional[int] = None,
    *,
    overwrite: bool = True,
    mrcfile_settings: Dict[str, Any] = None,
) -> RelionParticleStackDataset:
    if mrcfile_settings is None:
        mrcfile_settings = {
            "prefix": "downsampled",
            "delimiter": "_",
            "overwrite": overwrite,
        }
    if images_per_mrc is None:
        images_per_mrc = get_images_per_mrc(relion_dataset.parameter_file)

    # drop the rlnImageName column from the parameter file
    new_parameter_file = relion_dataset.parameter_file.copy()
    new_parameter_file.starfile_data["particles"].drop(
        "rlnImageName", axis=1, inplace=True
    )

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
    whitening_filter = _compute_whitening_filter(relion_dataset, dataloader)

    for batch in dataloader:
        whitened_images = irfftn(whitening_filter(rfftn(batch["stack"]["images"])))

        new_relion_dataset.append(
            {
                "images": whitened_images,
                "parameters": new_parameter_file[batch["index"]],
            }
        )

    new_relion_dataset.parameter_file.save(overwrite=True)
    jnp.save(
        os.path.join(path_to_new_relion_project, "whitening_filter.npy"),
        whitening_filter.array,
    )
    print(
        f"Whitening filter saved to {os.path.join(path_to_new_relion_project, 'whitening_filter.npy')}"
    )
    return new_relion_dataset


def _compute_whitening_filter(
    relion_dataset: RelionParticleStackDataset, dataloader: jdl.DataLoader
) -> op.CustomFilter:
    # compute whitening filter
    image = relion_dataset[0]["images"]
    whitening_filter = jnp.zeros(rfftn(image).shape, dtype=jnp.float32)

    for batch in dataloader:
        whitening_filter += op.WhiteningFilter(
            batch["stack"]["images"],
        ).array

    return op.CustomFilter(whitening_filter / len(dataloader))
