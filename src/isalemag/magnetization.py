"""Model remanence using a prescribed tracer history and blocking spectrum"""

import numpy as np
import pandas as pd


def _numeric_frame(frame, columns, name):
    """Validate finite numerical inputs without changing the caller's dataframe"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty dataframe")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
    result = frame.copy()
    if result.columns.duplicated().any():
        raise ValueError(f"{name} must have unique column names")
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}.{column} must contain finite values")
    return result


def magnetize_tracers(
    history,
    properties,
    blocking,
    field,
    pressure_half_pa=None,
    pressure_exponent=1.0,
):
    """Return remanence at each supplied tracer time, in a fixed Cartesian frame

    history requires tracer_id, time_s, and temperature_k
    properties requires tracer_id, initial_mx_a_m, initial_my_a_m,
    initial_mz_a_m, and trm_a_m_per_t
    blocking requires blocking_k and weight, with nonnegative weights summing to one
    field requires time_s, bx_t, by_t, and bz_t, covering the entire history

    heating to a bin's blocking temperature erases its remanence
    cooling below that temperature records the interpolated ambient field
    new remanence is weight * trm_a_m_per_t * field_vector_t
    the same thresholds describe blocking and unblocking in this approximation

    optional pressure loss uses exp(-ln(2) * (peak_pressure / pressure_half_pa)**n)
    the pressure driver is peak_pressure_pa when provided, otherwise pressure_pa
    increasing peak pressure attenuates the remanence present at that time
    cooling can acquire new remanence after the shock has passed
    this empirical pressure law represents one shock event, not repeated shocks
    """
    history = _numeric_frame(history, ["time_s", "temperature_k"], "history")
    properties = _numeric_frame(
        properties,
        ["initial_mx_a_m", "initial_my_a_m", "initial_mz_a_m", "trm_a_m_per_t"],
        "properties",
    )
    blocking = _numeric_frame(blocking, ["blocking_k", "weight"], "blocking")
    field = _numeric_frame(field, ["time_s", "bx_t", "by_t", "bz_t"], "field")

    for name, frame in [("history", history), ("properties", properties)]:
        if "tracer_id" not in frame or frame["tracer_id"].isna().any():
            raise ValueError(f"{name} requires nonmissing tracer_id values")
    if properties["tracer_id"].duplicated().any():
        raise ValueError("properties must have one row per tracer_id")
    if history.duplicated(["tracer_id", "time_s"]).any():
        raise ValueError("history must have one row per tracer_id and time_s")
    if not history["tracer_id"].isin(properties["tracer_id"]).all():
        raise ValueError("properties must cover every tracer_id in history")
    if (history["temperature_k"] <= 0).any() or (blocking["blocking_k"] <= 0).any():
        raise ValueError("temperature_k and blocking_k must be positive kelvin values")
    if (properties["trm_a_m_per_t"] < 0).any():
        raise ValueError("trm_a_m_per_t must be nonnegative")
    if (blocking["weight"] < 0).any() or not np.isclose(
        blocking["weight"].sum(), 1.0, rtol=0, atol=1e-10
    ):
        raise ValueError("blocking weights must be nonnegative and sum to one")

    # require explicit field coverage rather than extrapolating its direction
    field = field.sort_values("time_s")
    if field["time_s"].duplicated().any():
        raise ValueError("field.time_s must be unique")
    if (
        field["time_s"].iloc[0] > history["time_s"].min()
        or field["time_s"].iloc[-1] < history["time_s"].max()
    ):
        raise ValueError("field.time_s must cover all history times")

    pressure_column = "peak_pressure_pa" if "peak_pressure_pa" in history else "pressure_pa"
    if pressure_half_pa is not None:
        if not np.isfinite(pressure_half_pa) or pressure_half_pa <= 0:
            raise ValueError("pressure_half_pa must be finite and positive")
        if not np.isfinite(pressure_exponent) or pressure_exponent <= 0:
            raise ValueError("pressure_exponent must be finite and positive")
        history = _numeric_frame(history, [pressure_column], "history")
        if pressure_column == "peak_pressure_pa" and (history[pressure_column] < 0).any():
            raise ValueError("peak_pressure_pa must be nonnegative")

    properties = properties.set_index("tracer_id")
    thresholds = blocking["blocking_k"].to_numpy(dtype=float)
    weights = blocking["weight"].to_numpy(dtype=float)
    field_times = field["time_s"].to_numpy(dtype=float)
    field_vectors = field[["bx_t", "by_t", "bz_t"]].to_numpy(dtype=float)
    history = history.sort_values(["tracer_id", "time_s"]).reset_index(drop=True)
    inherited_output = np.zeros((len(history), 3))
    trm_output = np.zeros((len(history), 3))
    blocked_output = np.zeros(len(history))
    peak_output = np.zeros(len(history))
    retention_output = np.ones(len(history))

    for tracer_id, tracer in history.groupby("tracer_id", sort=False):
        material = properties.loc[tracer_id]
        initial = material[["initial_mx_a_m", "initial_my_a_m", "initial_mz_a_m"]].to_numpy(
            dtype=float
        )
        inherited = weights[:, None] * initial[None, :]
        trm = np.zeros_like(inherited)
        previous_temperature = None
        previous_time = None
        peak_pressure = 0.0
        previous_loss = 0.0

        for index, row in tracer.iterrows():
            temperature = float(row["temperature_k"])
            time = float(row["time_s"])

            if pressure_half_pa is not None:
                # apply each increment of peak pressure once, ignoring tensile pressure
                peak_pressure = max(peak_pressure, float(row[pressure_column]), 0.0)
                loss = np.log(2.0) * (peak_pressure / pressure_half_pa) ** pressure_exponent
                attenuation = np.exp(-(loss - previous_loss))
                inherited *= attenuation
                trm *= attenuation
                previous_loss = loss
                retention_output[index] = np.exp(-loss)

            unblocked = temperature >= thresholds
            inherited[unblocked] = 0.0
            trm[unblocked] = 0.0

            if previous_temperature is not None and temperature < previous_temperature:
                crossed = (previous_temperature >= thresholds) & (temperature < thresholds)
                if crossed.any():
                    fraction = (previous_temperature - thresholds[crossed]) / (
                        previous_temperature - temperature
                    )
                    crossing_times = previous_time + fraction * (time - previous_time)
                    crossing_field = np.column_stack(
                        [
                            np.interp(crossing_times, field_times, field_vectors[:, component])
                            for component in range(3)
                        ]
                    )
                    trm[crossed] = (
                        weights[crossed, None]
                        * float(material["trm_a_m_per_t"])
                        * crossing_field
                    )

            inherited_output[index] = inherited.sum(axis=0)
            trm_output[index] = trm.sum(axis=0)
            blocked_output[index] = weights[~unblocked].sum()
            peak_output[index] = peak_pressure
            previous_temperature = temperature
            previous_time = time

    result = history.copy()
    for component, axis in enumerate("xyz"):
        result[f"inherited_m{axis}_a_m"] = inherited_output[:, component]
        result[f"trm_m{axis}_a_m"] = trm_output[:, component]
        result[f"m{axis}_a_m"] = inherited_output[:, component] + trm_output[:, component]
    result["magnitude_a_m"] = np.linalg.norm(inherited_output + trm_output, axis=1)
    result["blocked_fraction"] = blocked_output
    if pressure_half_pa is not None:
        result["modeled_peak_pressure_pa"] = peak_output
        result["pressure_retention"] = retention_output
    return result
