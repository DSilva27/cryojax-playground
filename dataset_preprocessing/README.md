# Dependencies

The only dependencies are `cryojax` and `jax_dataloader`

```bash
pip install cryojax jax_dataloader
```

should install all the required dependencies.

# Running the scripts

The scripts in `runnables/` can be run as follows. For the downsampling script


```bash
PYTHONPATH=. python runnables/downsample_dataset.py path/to/starfile/mystarfile.star path/to/relion/project -D 2 --overwrite --images-per-mrc <int> --batch-size <int> -o path/to/output
```

Here `-D` is the downsampling factor, for 2 the box size will be halved. `--overwrite`,`images-per-mrc` and `--batch-size` are optional. The default `images-per-mrc` preserves the original dataset. `--bath-size` refers to the `batch_size` argument of `jax.lax.map`, reduce it if you run out of memory. If `--overwrite` is included, then the dataset will be overwritten, otherwise overwrite is set to `False`.


For the whitening script:
```bash
PYTHONPATH=. python runnables/whiten_dataset.py path/to/starfile/mystarfile.star path/to/relion/project --overwrite --images-per-mrc <int> -o path/to/output
```

See the arguments for the downsampling script for more information.

A greater degree of customization can be achieved by calling the functions in `src` yourself. See the `runnables` scripts for reference.
