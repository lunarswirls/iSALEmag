"""Convert pySALEPlot tracer histories to lowercase dataframe columns"""

import numpy as np
import pandas as pd


def read_isale_tracers(model, steps=None, tracer_ids=None):
    """Read a caller-owned pySALEPlot model without closing it or changing its scale

    Trt is instantaneous temperature in kelvin, not the peak-temperature field TrT
    TrP is cumulative peak pressure in pascals
    x_native and y_native retain the caller's pySALEPlot distance units
    positions in 2d may represent cylindrical radius and vertical position
    """
    if "Trt" not in model.plottype or "TrP" not in model.plottype:
        raise ValueError("iSALE output must save Trt and TrP in TR_VAR")
    if steps is None:
        steps = range(model.nsteps)
    if tracer_ids is None:
        tracer_ids = np.arange(model.tracer_num)
    tracer_ids = np.asarray(tracer_ids)
    if (
        tracer_ids.ndim != 1
        or tracer_ids.size == 0
        or not np.issubdtype(tracer_ids.dtype, np.integer)
        or (tracer_ids < 0).any()
        or (tracer_ids >= model.tracer_num).any()
        or np.unique(tracer_ids).size != tracer_ids.size
    ):
        raise ValueError("tracer_ids must contain unique valid integer tracer indices")

    frames = []
    for index in steps:
        step = model.readStep(["Trt", "TrP"], index)
        frame = pd.DataFrame(
            {
                "tracer_id": tracer_ids,
                "time_s": float(step.time),
                "temperature_k": np.asarray(step.data[0])[tracer_ids],
                "peak_pressure_pa": np.asarray(step.data[1])[tracer_ids],
                "x_native": np.asarray(step.xmark)[tracer_ids],
                "y_native": np.asarray(step.ymark)[tracer_ids],
            }
        )
        if getattr(model, "dimensions", 2) == 3:
            frame["z_native"] = np.asarray(step.zmark)[tracer_ids]
        frames.append(frame)
    if not frames:
        raise ValueError("steps must select at least one timestep")
    result = pd.concat(frames, ignore_index=True)
    result.attrs["position_units"] = "caller-selected pySALEPlot scale"
    return result
