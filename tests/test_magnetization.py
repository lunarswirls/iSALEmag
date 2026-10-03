import unittest

import numpy as np
import pandas as pd

from isalemag import magnetize_tracers


def make_inputs(temperatures, times=None):
    if times is None:
        times = np.arange(len(temperatures), dtype=float)
    history = pd.DataFrame({"tracer_id": 0, "time_s": times, "temperature_k": temperatures})
    properties = pd.DataFrame(
        {
            "tracer_id": [0],
            "initial_mx_a_m": [2.0],
            "initial_my_a_m": [0.0],
            "initial_mz_a_m": [0.0],
            "trm_a_m_per_t": [2.0e4],
        }
    )
    blocking = pd.DataFrame({"blocking_k": [400.0, 600.0], "weight": [0.25, 0.75]})
    field = pd.DataFrame({"time_s": [min(times), max(times)], "bx_t": 0, "by_t": 0, "bz_t": 50e-6})
    if len(temperatures) == 1:
        field = field.iloc[:1]
    return history, properties, blocking, field


class test_magnetization(unittest.TestCase):
    def test_cold_rock_preserves_initial_remanence(self):
        result = magnetize_tracers(*make_inputs([300, 320, 310]))
        np.testing.assert_allclose(result["mx_a_m"], 2.0)
        np.testing.assert_allclose(result["trm_mz_a_m"], 0.0)

    def test_full_reset_and_cooling(self):
        result = magnetize_tracers(*make_inputs([300, 700, 300]))
        np.testing.assert_allclose(result["mx_a_m"], [2, 0, 0])
        np.testing.assert_allclose(result["mz_a_m"], [0, 0, 1])
        np.testing.assert_allclose(result["blocked_fraction"], [1, 0, 1])

    def test_partial_reset_preserves_high_temperature_bins(self):
        result = magnetize_tracers(*make_inputs([300, 500, 300]))
        self.assertAlmostEqual(result.iloc[-1]["inherited_mx_a_m"], 1.5)
        self.assertAlmostEqual(result.iloc[-1]["trm_mz_a_m"], 0.25)

    def test_zero_field_cooling_does_not_restore_erased_remanence(self):
        history, properties, blocking, field = make_inputs([300, 700, 300])
        field["bz_t"] = 0.0
        result = magnetize_tracers(history, properties, blocking, field)
        self.assertEqual(result.iloc[-1]["magnitude_a_m"], 0.0)

    def test_hot_initial_state_cannot_preserve_unblocked_remanence(self):
        result = magnetize_tracers(*make_inputs([700, 300]))
        self.assertEqual(result.iloc[0]["magnitude_a_m"], 0.0)
        self.assertAlmostEqual(result.iloc[-1]["mz_a_m"], 1.0)

    def test_hot_final_state_does_not_acquire_future_trm(self):
        result = magnetize_tracers(*make_inputs([300, 700, 650]))
        self.assertEqual(result.iloc[-1]["magnitude_a_m"], 0.0)

    def test_field_is_sampled_at_each_blocking_crossing(self):
        history, properties, blocking, field = make_inputs([700, 300], [0, 4])
        field["bz_t"] = [50e-6, -50e-6]
        result = magnetize_tracers(history, properties, blocking, field)
        # the bins cross at 1 s and 3 s, recording opposite fields with unequal weights
        self.assertAlmostEqual(result.iloc[-1]["mz_a_m"], 0.25)

    def test_reheating_replaces_trm_instead_of_adding_it(self):
        history, properties, blocking, field = make_inputs([300, 700, 300, 700, 300])
        field = pd.DataFrame(
            {"time_s": [0, 2, 3, 4], "bx_t": 0, "by_t": 0, "bz_t": [50e-6, 50e-6, -50e-6, -50e-6]}
        )
        result = magnetize_tracers(history, properties, blocking, field)
        np.testing.assert_allclose(result["mz_a_m"], [0, 0, 1, 0, -1])

    def test_temperature_at_threshold_is_unblocked(self):
        result = magnetize_tracers(*make_inputs([300, 600, 599]))
        self.assertEqual(result.iloc[1]["blocked_fraction"], 0.0)
        self.assertAlmostEqual(result.iloc[2]["trm_mz_a_m"], 0.75)

    def test_pressure_applies_once_per_peak_increment(self):
        history, properties, blocking, field = make_inputs([300] * 5)
        history["pressure_pa"] = [0, 3e9, 3e9, 0, 6e9]
        result = magnetize_tracers(history, properties, blocking, field, pressure_half_pa=3e9)
        np.testing.assert_allclose(result["mx_a_m"], [2, 1, 1, 1, 0.5])
        np.testing.assert_allclose(result["pressure_retention"], [1, 0.5, 0.5, 0.5, 0.25])

    def test_peak_pressure_captures_unsampled_shock_and_ignores_tension(self):
        history, properties, blocking, field = make_inputs([300, 300])
        history["pressure_pa"] = [-1e9, -1e9]
        history["peak_pressure_pa"] = [0, 3e9]
        result = magnetize_tracers(history, properties, blocking, field, pressure_half_pa=3e9)
        self.assertAlmostEqual(result.iloc[-1]["mx_a_m"], 1.0)
        history = history.drop(columns="peak_pressure_pa")
        result = magnetize_tracers(history, properties, blocking, field, pressure_half_pa=3e9)
        self.assertAlmostEqual(result.iloc[-1]["mx_a_m"], 2.0)

    def test_cooling_trm_is_not_attenuated_by_a_past_shock(self):
        history, properties, blocking, field = make_inputs([300, 700, 300])
        history["peak_pressure_pa"] = [0, 30e9, 30e9]
        result = magnetize_tracers(history, properties, blocking, field, pressure_half_pa=3e9)
        self.assertAlmostEqual(result.iloc[-1]["mz_a_m"], 1.0)

    def test_vector_cancellation(self):
        history, properties, blocking, field = make_inputs([300, 500, 300])
        properties["initial_mx_a_m"] = 0.5
        field["bx_t"] = -75e-6
        field["bz_t"] = 0.0
        result = magnetize_tracers(history, properties, blocking, field)
        self.assertAlmostEqual(result.iloc[-1]["magnitude_a_m"], 0.0)

    def test_multiple_tracers_use_their_own_properties(self):
        history, properties, blocking, field = make_inputs([300, 700, 300])
        second_history = history.assign(tracer_id=10)
        second_properties = properties.assign(tracer_id=10, trm_a_m_per_t=4e4)
        combined = pd.concat([history, second_history]).sample(frac=1, random_state=0)
        combined_properties = pd.concat([properties, second_properties])
        original = combined.copy()
        result = magnetize_tracers(combined, combined_properties, blocking, field)
        final = result.groupby("tracer_id").tail(1).set_index("tracer_id")
        self.assertAlmostEqual(final.loc[0, "mz_a_m"], 1.0)
        self.assertAlmostEqual(final.loc[10, "mz_a_m"], 2.0)
        pd.testing.assert_frame_equal(combined, original)

    def test_invalid_inputs_raise(self):
        for changed in ["temperature", "weight", "field_coverage", "duplicate", "missing_property", "field_nan"]:
            with self.subTest(changed=changed):
                history, properties, blocking, field = make_inputs([300, 700, 300])
                if changed == "temperature":
                    history.loc[0, "temperature_k"] = np.nan
                elif changed == "weight":
                    blocking["weight"] = 1.0
                elif changed == "field_coverage":
                    field.loc[0, "time_s"] = 1
                elif changed == "duplicate":
                    history.loc[1, "time_s"] = 0
                elif changed == "missing_property":
                    properties["tracer_id"] = 1
                elif changed == "field_nan":
                    field.loc[0, "bz_t"] = np.nan
                with self.assertRaises(ValueError):
                    magnetize_tracers(history, properties, blocking, field)

    def test_pressure_requires_a_valid_scale_and_driver(self):
        for pressure_half_pa in [0, -1, np.inf, 3e9]:
            with self.subTest(pressure_half_pa=pressure_half_pa):
                with self.assertRaises(ValueError):
                    magnetize_tracers(*make_inputs([300, 300]), pressure_half_pa=pressure_half_pa)
