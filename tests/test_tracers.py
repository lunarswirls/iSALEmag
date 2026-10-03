from types import SimpleNamespace
import unittest

import numpy as np

from isalemag import read_isale_tracers


class test_tracers(unittest.TestCase):
    def setUp(self):
        def read_step(fields, index):
            self.assertEqual(fields, ["Trt", "TrP"])
            return SimpleNamespace(
                time=float(index),
                data=[np.array([300, 700]) + index, np.array([0, 3e9])],
                xmark=np.array([1.0, 2.0]),
                ymark=np.array([-1.0, -2.0]),
                zmark=np.array([0.0, 0.0]),
            )

        self.model = SimpleNamespace(plottype=["Trt", "TrP"], tracer_num=2, nsteps=2, dimensions=2)
        setattr(self.model, "readStep", read_step)

    def test_extracts_current_temperature_and_peak_pressure(self):
        result = read_isale_tracers(self.model, tracer_ids=[1])
        np.testing.assert_allclose(result["temperature_k"], [700, 701])
        np.testing.assert_allclose(result["peak_pressure_pa"], 3e9)
        self.assertEqual(result["tracer_id"].tolist(), [1, 1])
        self.assertNotIn("z_native", result)

    def test_three_dimensional_positions(self):
        self.model.dimensions = 3
        result = read_isale_tracers(self.model, steps=[1])
        self.assertIn("z_native", result)

    def test_peak_temperature_cannot_replace_current_temperature(self):
        self.model.plottype = ["TrT", "TrP"]
        with self.assertRaisesRegex(ValueError, "Trt"):
            read_isale_tracers(self.model)

    def test_invalid_indices_and_empty_steps(self):
        for tracer_ids in [[2], [-1], [0.5], [0, 0], []]:
            with self.subTest(tracer_ids=tracer_ids):
                with self.assertRaises(ValueError):
                    read_isale_tracers(self.model, tracer_ids=tracer_ids)
        with self.assertRaises(ValueError):
            read_isale_tracers(self.model, steps=[])
