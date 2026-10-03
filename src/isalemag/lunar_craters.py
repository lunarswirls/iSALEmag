"""Axisymmetric geometry helpers for a lunar central-peak sensitivity experiment"""

import numpy as np
import pandas as pd

from .magnetization import _numeric_frame


def lunar_crater_scaling(cases):
    """Return reference rim-to-floor depths and peak relief in kilometers

    the power laws are the mare and highlands fits in kalynn et al (2013), table 1
    diameter_km is rim-to-rim diameter and peak height is measured above the floor
    these reference fits do not determine pressure, temperature, or uplift history
    """
    cases = _numeric_frame(cases, ["diameter_km"], "cases")
    if "terrain" not in cases or not cases["terrain"].isin(["mare", "highlands"]).all():
        raise ValueError("terrain must be mare or highlands")
    if (cases["diameter_km"] <= 0).any():
        raise ValueError("diameter_km must be positive")
    coefficients = pd.DataFrame(
        [
            ["mare", 0.870, 0.352, 0.075, 0.614],
            ["highlands", 1.558, 0.254, 0.034, 0.883],
        ],
        columns=["terrain", "depth_coefficient", "depth_exponent", "peak_coefficient", "peak_exponent"],
    )
    result = cases.merge(coefficients, on="terrain", validate="many_to_one")
    result["reference_depth_km"] = result["depth_coefficient"] * result["diameter_km"] ** result["depth_exponent"]
    result["reference_peak_height_km"] = result["peak_coefficient"] * result["diameter_km"] ** result["peak_exponent"]
    result["reference_depth_diameter_ratio"] = result["reference_depth_km"] / result["diameter_km"]
    return result


def axisymmetric_source_cells(
    diameter_m,
    radial_cells=48,
    depth_cells=96,
    source_depth_fraction=0.40,
    radial_breaks=(0.10, 0.20, 0.30),
):
    """Return annular source cells beneath the original flat surface

    radial_breaks are fractions of crater diameter that split the radial mesh
    source depths are positive downward from the original surface and rim datum
    """
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("diameter_m must be finite and positive")
    if not np.isfinite(source_depth_fraction) or source_depth_fraction <= 0:
        raise ValueError("source_depth_fraction must be finite and positive")
    for count in [radial_cells, depth_cells]:
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("radial_cells and depth_cells must be positive integers")
    radial_breaks = np.asarray(radial_breaks, dtype=float)
    if radial_breaks.ndim != 1 or not np.isfinite(radial_breaks).all():
        raise ValueError("radial_breaks must be a finite one-dimensional sequence")
    if ((radial_breaks <= 0) | (radial_breaks >= 0.5)).any():
        raise ValueError("radial_breaks must lie strictly between zero and 0.5")
    radius_edges = np.unique(
        np.concatenate((np.linspace(0, diameter_m / 2, radial_cells + 1), radial_breaks * diameter_m))
    )
    depth_edges = np.linspace(0, source_depth_fraction * diameter_m, depth_cells + 1)
    radius_inner, source_top = np.meshgrid(radius_edges[:-1], depth_edges[:-1])
    radius_outer, source_bottom = np.meshgrid(radius_edges[1:], depth_edges[1:])
    result = pd.DataFrame(
        {
            "tracer_id": np.arange(radius_inner.size),
            "radius_inner_m": radius_inner.ravel(),
            "radius_outer_m": radius_outer.ravel(),
            "source_top_m": source_top.ravel(),
            "source_bottom_m": source_bottom.ravel(),
        }
    )
    # the midpoint in squared radius integrates the paraboloid peak relief exactly
    result["radius_m"] = np.sqrt((result["radius_inner_m"] ** 2 + result["radius_outer_m"] ** 2) / 2)
    result["source_depth_m"] = (result["source_top_m"] + result["source_bottom_m"]) / 2
    result["source_volume_m3"] = np.pi * (
        result["radius_outer_m"] ** 2 - result["radius_inner_m"] ** 2
    ) * (result["source_bottom_m"] - result["source_top_m"])
    return result


def central_peak_geometry(
    cells,
    diameter_m,
    depth_diameter_ratio,
    peak_height_m,
    peak_radius_fraction=0.10,
    floor_radius_fraction=0.30,
    uplift_radius_fraction=0.20,
    uplift_depth_factor=2.0,
    floor_layer_fraction=0.02,
):
    """Map source cells into a flat-floor crater with a paraboloid central peak

    d is rim-to-floor depth, excluding peak relief, and D is rim-to-rim diameter
    uplift is a prescribed vertical translation with amplitude uplift_depth_factor*d
    the translation preserves annular cell volume before excavation
    peak_volume_m3 includes only relief above the nominal floor, excluding roots
    floor_layer_volume_m3 samples a fixed-thickness layer beneath the exposed floor
    geometry and uplift are parameterized assumptions rather than a crater simulation
    """
    cells = _numeric_frame(
        cells,
        ["radius_inner_m", "radius_outer_m", "source_top_m", "source_bottom_m"],
        "cells",
    )
    parameters = np.asarray(
        [diameter_m, depth_diameter_ratio, peak_height_m, peak_radius_fraction,
         floor_radius_fraction, uplift_radius_fraction, uplift_depth_factor, floor_layer_fraction],
        dtype=float,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("geometry parameters must be finite")
    if diameter_m <= 0 or depth_diameter_ratio <= 0 or floor_layer_fraction <= 0:
        raise ValueError("diameter, depth ratio, and floor layer fraction must be positive")
    crater_depth_m = depth_diameter_ratio * diameter_m
    if not 0 < peak_height_m < crater_depth_m:
        raise ValueError("peak_height_m must be positive and below the rim-to-floor depth")
    if not 0 < peak_radius_fraction <= uplift_radius_fraction <= floor_radius_fraction < 0.5:
        raise ValueError("require 0 < peak radius <= uplift radius <= floor radius < 0.5 diameter")
    if uplift_depth_factor < 0:
        raise ValueError("uplift_depth_factor must be nonnegative")
    if (
        (cells["radius_inner_m"] < 0).any()
        or (cells["radius_outer_m"] <= cells["radius_inner_m"]).any()
        or (cells["radius_outer_m"] > diameter_m / 2).any()
        or (cells["source_top_m"] < 0).any()
        or (cells["source_bottom_m"] <= cells["source_top_m"]).any()
    ):
        raise ValueError("cells require ordered positive-width bounds inside the crater radius")
    peak_radius_m = peak_radius_fraction * diameter_m
    floor_radius_m = floor_radius_fraction * diameter_m
    for boundary in [peak_radius_m, floor_radius_m]:
        if ((cells["radius_inner_m"] < boundary) & (cells["radius_outer_m"] > boundary)).any():
            raise ValueError("the radial mesh must split at the peak and floor radius boundaries")

    radius = np.sqrt((cells["radius_inner_m"] ** 2 + cells["radius_outer_m"] ** 2) / 2)
    ring_area = np.pi * (cells["radius_outer_m"] ** 2 - cells["radius_inner_m"] ** 2)
    thickness = cells["source_bottom_m"] - cells["source_top_m"]
    bowl_depth = crater_depth_m * np.clip(
        (diameter_m / 2 - radius) / (diameter_m / 2 - floor_radius_m), 0, 1
    )
    peak_relief = peak_height_m * np.clip(1 - (radius / peak_radius_m) ** 2, 0, 1)
    surface_depth = bowl_depth - peak_relief
    uplift = uplift_depth_factor * crater_depth_m * np.clip(
        1 - (radius / (uplift_radius_fraction * diameter_m)) ** 2, 0, 1
    ) ** 2
    retained_top = np.maximum(cells["source_top_m"], surface_depth + uplift)
    result = cells.copy()
    result["radius_m"] = radius
    result["source_depth_m"] = (cells["source_top_m"] + cells["source_bottom_m"]) / 2
    result["surface_depth_m"] = surface_depth
    result["uplift_m"] = uplift
    result["final_top_m"] = cells["source_top_m"] - uplift
    result["final_bottom_m"] = cells["source_bottom_m"] - uplift
    result["final_depth_m"] = result["source_depth_m"] - uplift
    result["retained_volume_m3"] = ring_area * np.clip(cells["source_bottom_m"] - retained_top, 0, thickness)
    result["peak_volume_m3"] = ring_area * np.clip(
        np.minimum(cells["source_bottom_m"], crater_depth_m + uplift) - retained_top, 0, thickness
    ) * (radius < peak_radius_m)
    result["peak_source_depth_m"] = (
        retained_top + np.minimum(cells["source_bottom_m"], crater_depth_m + uplift)
    ) / 2
    floor_top = np.maximum(cells["source_top_m"], crater_depth_m + uplift)
    floor_bottom = np.minimum(cells["source_bottom_m"], crater_depth_m + uplift + floor_layer_fraction * diameter_m)
    result["floor_layer_volume_m3"] = ring_area * np.clip(floor_bottom - floor_top, 0, thickness) * (
        (radius >= peak_radius_m) & (radius < floor_radius_m)
    )
    result["floor_layer_source_depth_m"] = (floor_top + floor_bottom) / 2
    expected_peak_volume = np.pi * peak_radius_m**2 * peak_height_m / 2
    if not np.isclose(result["peak_volume_m3"].sum(), expected_peak_volume, rtol=1e-10):
        raise ValueError("the source mesh does not cover the full central peak volume")
    expected_floor_volume = np.pi * (floor_radius_m**2 - peak_radius_m**2) * floor_layer_fraction * diameter_m
    if not np.isclose(result["floor_layer_volume_m3"].sum(), expected_floor_volume, rtol=1e-10):
        raise ValueError("the source mesh does not cover the full floor sampling layer")
    return result
