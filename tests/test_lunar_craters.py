import unittest

import numpy as np
import pandas as pd

from isalemag.lunar_craters import axisymmetric_source_cells, central_peak_geometry, lunar_crater_scaling


class test_lunar_craters(unittest.TestCase):
    def test_reference_scaling_preserves_terrain_distinction(self):
        cases = pd.DataFrame({"diameter_km": [75.0, 75.0], "terrain": ["mare", "highlands"]})
        original = cases.copy()
        result = lunar_crater_scaling(cases).set_index("terrain")
        self.assertGreater(result.loc["highlands", "reference_depth_km"], result.loc["mare", "reference_depth_km"])
        self.assertGreater(result.loc["highlands", "reference_peak_height_km"], result.loc["mare", "reference_peak_height_km"])
        np.testing.assert_allclose(result["reference_depth_diameter_ratio"], result["reference_depth_km"] / 75)
        pd.testing.assert_frame_equal(cases, original)

    def test_annular_source_volume_matches_cylinder(self):
        cells = axisymmetric_source_cells(75000, radial_cells=13, depth_cells=17)
        expected_volume = np.pi * 37500**2 * 30000
        self.assertAlmostEqual(cells["source_volume_m3"].sum() / expected_volume, 1.0)

    def test_peak_and_floor_volumes_are_resolution_independent(self):
        diameter_m = 75000
        peak_height_m = 1500
        expected_peak = np.pi * 7500**2 * peak_height_m / 2
        expected_floor = np.pi * (22500**2 - 7500**2) * 1500
        for radial_cells, depth_cells in [(12, 16), (48, 96)]:
            cells = axisymmetric_source_cells(diameter_m, radial_cells=radial_cells, depth_cells=depth_cells)
            for ratio in [0.04, 0.07, 0.10]:
                with self.subTest(radial_cells=radial_cells, depth_cells=depth_cells, ratio=ratio):
                    mapped = central_peak_geometry(cells, diameter_m, ratio, peak_height_m)
                    self.assertAlmostEqual(mapped["peak_volume_m3"].sum() / expected_peak, 1.0)
                    self.assertAlmostEqual(mapped["floor_layer_volume_m3"].sum() / expected_floor, 1.0)
                    self.assertTrue((mapped["peak_volume_m3"] <= mapped["retained_volume_m3"] + 1e-3).all())

    def test_peak_thinner_than_a_cell_is_preserved(self):
        cells = axisymmetric_source_cells(75000, radial_cells=12, depth_cells=8)
        mapped = central_peak_geometry(cells, 75000, 0.07, 40)
        expected_peak = np.pi * 7500**2 * 40 / 2
        self.assertAlmostEqual(mapped["peak_volume_m3"].sum() / expected_peak, 1.0)
        peak_cells = mapped.loc[mapped["peak_volume_m3"] > 0]
        self.assertTrue((peak_cells["peak_volume_m3"] < peak_cells["source_volume_m3"]).all())

    def test_vertical_translation_preserves_material_volume(self):
        cells = axisymmetric_source_cells(75000, radial_cells=12, depth_cells=16)
        mapped = central_peak_geometry(cells, 75000, 0.07, 1500)
        np.testing.assert_allclose(
            mapped["final_bottom_m"] - mapped["final_top_m"], cells["source_bottom_m"] - cells["source_top_m"]
        )
        self.assertTrue((mapped["retained_volume_m3"] <= mapped["source_volume_m3"] + 1e-3).all())
        self.assertTrue((mapped["retained_volume_m3"] < mapped["source_volume_m3"]).any())

    def test_increasing_depth_samples_deeper_peak_material(self):
        cells = axisymmetric_source_cells(75000, radial_cells=12, depth_cells=16)
        mean_depths = []
        for ratio in [0.04, 0.10]:
            mapped = central_peak_geometry(cells, 75000, ratio, 1500)
            weights = mapped["peak_volume_m3"] / mapped["peak_volume_m3"].sum()
            mean_depths.append((mapped["peak_source_depth_m"] * weights).sum())
        self.assertGreater(mean_depths[1], mean_depths[0])

    def test_missing_peak_source_material_is_rejected(self):
        cells = axisymmetric_source_cells(75000, source_depth_fraction=0.03)
        with self.assertRaisesRegex(ValueError, "full central peak"):
            central_peak_geometry(cells, 75000, 0.07, 1500)

    def test_radius_boundaries_cannot_cut_unresolved_rings(self):
        cells = axisymmetric_source_cells(75000, radial_cells=11, radial_breaks=[])
        with self.assertRaisesRegex(ValueError, "split at"):
            central_peak_geometry(cells, 75000, 0.07, 1500)

    def test_invalid_reference_and_geometry_inputs(self):
        with self.assertRaises(ValueError):
            lunar_crater_scaling(pd.DataFrame({"diameter_km": [75], "terrain": ["unknown"]}))
        with self.assertRaises(ValueError):
            axisymmetric_source_cells(75000, depth_cells=1.5)
        cells = axisymmetric_source_cells(75000, radial_cells=12, depth_cells=16)
        for ratio, height in [(0, 1500), (np.nan, 1500), (0.04, 5000)]:
            with self.subTest(ratio=ratio, height=height):
                with self.assertRaises(ValueError):
                    central_peak_geometry(cells, 75000, ratio, height)
