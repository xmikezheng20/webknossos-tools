# webknossos-tools

Convert TIFF stacks into WEBKNOSSOS datasets with multiple image layers and
resolution pyramids. BrainGlobe determines orientation mapping; the WEBKNOSSOS
Python API handles import and downsampling.

## Environment

Create the Conda environment once:

```bash
conda create --prefix ~/.conda/envs/webknossos -c conda-forge python=3.12
```

From the repository root, activate it and install the dependencies:

```bash
conda activate webknossos
python -m pip install -r requirements.txt
```

Use the same setup in the production checkout under the `webknossos` user.

## Convert

Run from the repository root. First inspect TIFF headers and the proposed
orientation and spacing without creating an output dataset:

```bash
python scripts/convert.py recipes/stpt_example.yaml --check
```

Then convert:

```bash
python scripts/convert.py recipes/stpt_example.yaml
```

Each layer glob selects an already aligned stack of single-page 2D uint16 TIFFs.
Files are naturally sorted, so `section_2.tif` precedes `section_10.tif`.
All layers must have matching plane counts and image dimensions; corresponding
sorted files must represent corresponding planes. The script checks headers
but does not perform image registration.

Layers are imported sequentially with `jobs` workers (default: 2). Import uses
WEBKNOSSOS CLI defaults: compressed Zarr v3, native input dtype, and automatic
batching and storage layout. Downsampling builds a compressed anisotropic
pyramid using the default median filter for image layers. Full-resolution
uint16 values are preserved; coarser levels contain filtered values.

Conversion writes to `<output>.partial`, saves the recipe as `recipe.yaml`, then
renames the directory to `<output>` after success. Existing output and partial
directories are refused. A failed run leaves the partial directory for
inspection; remove it or choose another output before rerunning.

## Recipe and coordinates

`recipes/stpt_example.yaml` describes two STPT channels as separate image layers.
Input array axes are `(slice, TIFF row, TIFF column)`. The three letters in
`input_orientation` and the three values in `input_voxel_size_um` follow that
same order. BrainGlobe letters identify the anatomical side at index zero.
For `ars`, increasing indices run:

| Input array axis | Anatomical direction | Example spacing, um |
| --- | --- | --- |
| Slice | Anterior to posterior | 50 |
| TIFF row | Right to left | 1 |
| TIFF column | Superior to inferior | 1 |

The example reorients `ars` to `asr`: swap row and column axes, keeping the slice
axis. Voxel spacing follows the same permutation. WEBKNOSSOS expects XYZ, so
the script reverses the target array's axis order when specifying spacing.
For example, input `ars` spacing `[50, 2, 3]` becomes target `asr` array spacing
`[50, 3, 2]` and WEBKNOSSOS XYZ spacing `[2, 3, 50]`.

Omitting `output_orientation` preserves native orientation; removing a layer
imports only the remaining channel.

Reorientation uses WEBKNOSSOS's TIFF import options: `swap_xy`, `flip_x`,
`flip_y`, and `flip_z`. They cover 16 of the 48 possible combinations of 3D
axis permutations and flips: two in-plane axis orders times eight combinations
of flips. The original slice axis must remain the output slice axis, although
its direction can reverse. The importer has no `swap_xz` or `swap_yz` option.

For input `ars`, targets `asr` and `pir` are supported. Targets `sar` and `ras`
are rejected because they move an original in-plane axis into the slice axis.
Supporting those orientations would require an additional 3D transpose step.
Supported reorientation preserves values exactly, without resampling.
Arbitrary-angle rotations, scaling, and registration require separate processing.

Relative `source_root` and `output` paths resolve from the working directory.
Layer globs resolve under `source_root`. The dataset name is the basename of
`output`. The example writes under `data/`, which Git ignores. For production,
set `output` to a dataset path under `/mnt/hdd1/webknossos_data/`.

## Verify

```bash
python -m unittest discover -s tests -v
```

The integration tests convert small non-square LZW TIFF stacks with unequal
voxel spacing. They compare all 16 supported orientations against BrainGlobe,
check exact uint16 values and spacing, and verify compression and pyramids.
