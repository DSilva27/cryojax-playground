from cryojax.data import RelionParticleParameterFile, RelionParticleStackDataset
import jax_dataloader as jdl
from typing import Dict

# noqa: F722


class CustomJaxDataset(jdl.Dataset):
    cryojax_dataset: RelionParticleStackDataset

    def __init__(
        self,
        cryojax_dataset: RelionParticleStackDataset,
    ):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> Dict:
        return {
            "stack": self.cryojax_dataset[index],
            "index": index,
        }

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


def get_images_per_mrc(parameter_file: RelionParticleParameterFile) -> int:
    new_df = parameter_file.starfile_data["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    new_df.columns = ["index", "filename"]
    new_df["index"] = new_df["index"].astype(int)

    # group by unique filenames
    grouped = new_df.groupby("filename")["index"].apply(list).reset_index()
    return int(grouped["index"].apply(len).max())


def create_dataloader(
    relion_dataset: RelionParticleStackDataset, batch_size
) -> jdl.DataLoader:
    return jdl.DataLoader(
        CustomJaxDataset(relion_dataset),
        backend="jax",
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
