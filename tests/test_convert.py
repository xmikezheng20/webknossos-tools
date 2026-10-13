"""Integration checks for TIFF orientation, spacing, and conversion defaults."""

from contextlib import chdir
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from brainglobe_space import AnatomicalSpace
import numpy as np
import tifffile
import webknossos as wk
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert.py"
spec = importlib.util.spec_from_file_location("convert_tiffs", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="webknossos-convert-test-")
        cls.root = Path(cls.temporary.name)
        z, row, column = np.indices((64, 40, 56), dtype=np.uint16)
        red = z * 900 + row * 13 + column
        red[0, 0, 0] = 65535
        red[-1, -1, -1] = 0
        cls.volumes = {"red": red, "green": np.uint16(65535) - red}
        for name, volume in cls.volumes.items():
            folder = cls.root / "inputs" / name
            folder.mkdir(parents=True)
            for index, plane in enumerate(volume, 1):
                # Unpadded numbers exercise natural filename sorting.
                tifffile.imwrite(folder / f"section_{index}.tif", plane, compression="lzw")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def recipe(self, **changes):
        recipe = {
            "source_root": "inputs",
            "output": "outputs/dataset",
            "input_orientation": "ars",
            "input_voxel_size_um": [50, 2, 3],
            "layers": {"red": "red/*.tif", "green": "green/*.tif"},
            "jobs": 2,
        }
        recipe.update(changes)
        path = self.root / "recipe.yaml"
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        return path

    def test_all_supported_orientations(self):
        # Both in-plane axis orders, with every combination of axis flips.
        targets = [a + b + c for a in "ap" for b in "rl" for c in "si"]
        targets += [a + b + c for a in "ap" for b in "si" for c in "rl"]
        spacing_by_direction = {"a": 50, "p": 50, "r": 2, "l": 2, "s": 3, "i": 3}
        source = AnatomicalSpace("ars")
        with chdir(self.root):
            for target in targets:
                with self.subTest(orientation=target):
                    changes = {"output": f"outputs/{target}"}
                    if target == "ars":
                        # Also test a single layer and omitted output orientation.
                        changes["layers"] = {"red": "red/*.tif"}
                    else:
                        changes["output_orientation"] = target
                    recipe_path = self.recipe(**changes)
                    recipe = converter.load_recipe(recipe_path)
                    order, flags, spacing = converter.plan_orientation(recipe)
                    layers, shape = converter.inspect_layers(recipe)
                    self.assertEqual(shape, (64, 40, 56))
                    dataset = converter.convert(recipe, recipe_path, layers, flags, spacing)

                    expected_spacing = tuple(spacing_by_direction[axis] for axis in target[::-1])
                    self.assertEqual(tuple(dataset.voxel_size_with_unit.factor), expected_spacing)
                    self.assertEqual(dataset.voxel_size_with_unit.unit, wk.LengthUnit.MICROMETER)
                    self.assertEqual(tuple(dataset.voxel_size), tuple(v * 1000 for v in expected_spacing))
                    self.assertEqual(set(dataset.layers), set(layers))
                    for name, layer in dataset.layers.items():
                        expected = source.map_stack_to(target, self.volumes[name])
                        actual = layer.get_mag(1).read_cxyz()[0].transpose(2, 1, 0)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, np.dtype("uint16"))
                        self.assertEqual(layer.data_format, wk.DataFormat.Zarr3)
                        self.assertGreater(len(layer.mags), 1)
                        for magnification, view in layer.mags.items():
                            self.assertTrue(view.info.compression_mode)
                            self.assertEqual(magnification.z, 1)
                            self.assertEqual(view.read_cxyz().dtype, np.dtype("uint16"))
                        first_coarse = next(mag for mag in layer.mags if mag != wk.Mag(1))
                        block = expected[:first_coarse.z, :first_coarse.y, :first_coarse.x]
                        self.assertEqual(
                            layer.get_mag(first_coarse).read_cxyz()[0, 0, 0, 0],
                            np.uint16(np.median(block)),
                        )
                    output = recipe["output"]
                    self.assertFalse(output.with_name(output.name + ".partial").exists())
                    self.assertEqual((output / "recipe.yaml").read_text(), recipe_path.read_text())

    def test_check_only(self):
        recipe_path = self.recipe(output="check_only/dataset", output_orientation="asr")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(recipe_path), "--check"],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        self.assertIn("WEBKNOSSOS XYZ spacing, um: (2.0, 3.0, 50.0)", result.stdout)
        self.assertIn("Check complete; no dataset written.", result.stdout)
        self.assertFalse((self.root / "check_only").exists())

    def test_mismatched_plane_counts(self):
        with chdir(self.root):
            path = self.recipe(layers={"red": "red/*.tif", "green": "green/section_1.tif"})
            recipe = converter.load_recipe(path)
            with self.assertRaisesRegex(ValueError, "expected 64 planes, found 1"):
                converter.inspect_layers(recipe)

    def test_slice_axis_permutation_is_rejected(self):
        with chdir(self.root):
            recipe = converter.load_recipe(self.recipe(output_orientation="sar"))
            with self.assertRaisesRegex(ValueError, "retain the slice axis"):
                converter.plan_orientation(recipe)

    def test_existing_partial_is_rejected(self):
        partial = self.root / "existing.partial"
        partial.mkdir()
        with chdir(self.root):
            with self.assertRaisesRegex(FileExistsError, "Output already exists"):
                converter.load_recipe(self.recipe(output="existing"))


if __name__ == "__main__":
    unittest.main()
