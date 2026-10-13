#!/usr/bin/env python3
"""Convert recipe-defined TIFF stacks into a WEBKNOSSOS dataset."""

import argparse
from importlib.metadata import version
from math import isfinite
from pathlib import Path
import shutil
from time import monotonic

from brainglobe_space import AnatomicalSpace
from cluster_tools import get_executor
from natsort import natsorted
import tifffile
import webknossos as wk
import yaml


def load_recipe(path: Path) -> dict:
    with path.open() as stream:
        recipe = yaml.safe_load(stream)
    if not isinstance(recipe, dict):
        raise ValueError("The recipe must be a YAML mapping.")
    for key in ("source_root", "output", "input_orientation", "input_voxel_size_um", "layers"):
        if key not in recipe:
            raise ValueError(f"Missing recipe field: {key}")

    recipe["source_root"] = Path(recipe["source_root"]).expanduser().resolve()
    recipe["output"] = Path(recipe["output"]).expanduser().resolve()
    recipe.setdefault("output_orientation", recipe["input_orientation"])
    recipe.setdefault("jobs", 2)
    recipe["input_voxel_size_um"] = tuple(float(v) for v in recipe["input_voxel_size_um"])
    spacing = recipe["input_voxel_size_um"]
    if len(spacing) != 3 or not all(isfinite(v) and v > 0 for v in spacing):
        raise ValueError("input_voxel_size_um must contain three positive, finite values.")
    if type(recipe["jobs"]) is not int or recipe["jobs"] < 1:
        raise ValueError("jobs must be a positive integer.")
    if not isinstance(recipe["layers"], dict) or not recipe["layers"]:
        raise ValueError("layers must map at least one layer name to a TIFF glob.")
    if not recipe["source_root"].is_dir():
        raise ValueError(f"Source directory does not exist: {recipe['source_root']}")
    output = recipe["output"]
    for path in (output, output.with_name(output.name + ".partial")):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    return recipe


def inspect_layers(recipe: dict) -> tuple[dict[str, list[Path]], tuple[int, int, int]]:
    """Inspect headers only; corresponding sorted files must be aligned planes."""
    layers = {}
    plane_shape = None
    plane_count = None
    for name, pattern in recipe["layers"].items():
        files = natsorted(recipe["source_root"].glob(pattern), key=str)
        if not files:
            raise ValueError(f"Layer {name!r}: no files match {pattern!r}.")
        if plane_count is None:
            plane_count = len(files)
        elif len(files) != plane_count:
            raise ValueError(f"Layer {name!r}: expected {plane_count} planes, found {len(files)}.")
        for path in files:
            with tifffile.TiffFile(path) as tiff:
                page = tiff.pages[0]
                if len(tiff.pages) != 1 or len(page.shape) != 2:
                    raise ValueError(f"Expected one 2D grayscale plane: {path}")
                if page.dtype.name != "uint16":
                    raise ValueError(f"Expected uint16, found {page.dtype}: {path}")
                if plane_shape is None:
                    plane_shape = page.shape
                elif page.shape != plane_shape:
                    raise ValueError(f"Expected TIFF shape {plane_shape}, found {page.shape}: {path}")
        layers[name] = files
    return layers, (plane_count, *plane_shape)


def plan_orientation(recipe: dict) -> tuple[tuple, dict, tuple]:
    """Return the array permutation, WEBKNOSSOS flags, and XYZ voxel spacing."""
    source = AnatomicalSpace(recipe["input_orientation"])
    target = AnatomicalSpace(recipe["output_orientation"])
    order, flips, _, _ = source.map_to(target)
    if order[0] != 0:
        raise ValueError("Reorientation must retain the slice axis as the first array axis.")

    # BrainGlobe's flips refer to target array axes. WEBKNOSSOS 4.2.4 flips
    # input TIFF rows (flip_x) and columns (flip_y) BEFORE its XY transpose.
    source_flips = tuple(flips[order.index(axis)] for axis in range(3))
    flags = {
        "swap_xy": order == (0, 2, 1),
        "flip_z": source_flips[0],
        "flip_x": source_flips[1],
        "flip_y": source_flips[2],
    }
    # Input and target arrays are (slice, row, column); WEBKNOSSOS is XYZ.
    target_spacing = tuple(recipe["input_voxel_size_um"][axis] for axis in order)
    return order, flags, target_spacing[::-1]


def convert(
    recipe: dict, recipe_path: Path, layers: dict, flags: dict, spacing_xyz: tuple
) -> wk.Dataset:
    output = recipe["output"]
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with get_executor("multiprocessing", max_workers=recipe["jobs"]) as executor:
        dataset = wk.Dataset(
            partial,
            name=output.name,
            voxel_size_with_unit=wk.VoxelSize(spacing_xyz, wk.LengthUnit.MICROMETER),
        )
        for name, files in layers.items():
            print(f"Importing layer {name}: {len(files)} planes", flush=True)
            dataset.add_layer_from_images(files, name, executor=executor, **flags)
        print("Building anisotropic pyramids", flush=True)
        dataset.downsample(executor=executor)

    shutil.copyfile(recipe_path, partial / "recipe.yaml")
    partial.rename(output)
    return wk.Dataset.open(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="Path to the conversion recipe.")
    parser.add_argument(
        "--check", action="store_true",
        help="Inspect inputs and print the plan without writing a dataset.",
    )
    args = parser.parse_args()
    started = monotonic()
    recipe_path = args.recipe.expanduser().resolve()
    recipe = load_recipe(recipe_path)
    order, flags, spacing_xyz = plan_orientation(recipe)
    print("Inspecting TIFF headers", flush=True)
    layers, input_shape = inspect_layers(recipe)
    target_shape = tuple(input_shape[axis] for axis in order)

    print(f"Source: {recipe['source_root']}")
    print(f"Output: {recipe['output']}")
    print(f"Array orientation: {recipe['input_orientation']} -> {recipe['output_orientation']}")
    print(f"Array shape (slice, row, column): {input_shape} -> {target_shape}")
    print(f"Input spacing (slice, row, column), um: {recipe['input_voxel_size_um']}")
    print(f"WEBKNOSSOS XYZ spacing, um: {spacing_xyz}")
    print(f"Import flags: {flags}")
    print(f"Workers: {recipe['jobs']}")
    packages = ("webknossos", "brainglobe-space", "PyYAML")
    print("Versions: " + ", ".join(f"{name}={version(name)}" for name in packages))
    for name, files in layers.items():
        print(f"Layer {name}: {len(files)} uint16 planes; first={files[0].name}; last={files[-1].name}")
    if args.check:
        print("Check complete; no dataset written.")
        return

    dataset = convert(recipe, recipe_path, layers, flags, spacing_xyz)
    print(f"Completed: {recipe['output']}")
    for name, layer in dataset.layers.items():
        print(f"Layer {name} magnifications: {', '.join(str(mag) for mag in layer.mags)}")
    print(f"Elapsed: {monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
