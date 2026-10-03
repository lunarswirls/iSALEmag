"""Test depth-to-diameter sensitivity of lunar central-peak magnetization"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from isalemag import magnetize_tracers
from isalemag.lunar_craters import axisymmetric_source_cells, central_peak_geometry, lunar_crater_scaling


def summarize_region(cells, volume_column):
    """Return volume-weighted vectors and moments for a specified material region"""
    volume = cells[volume_column].sum()
    if volume <= 0:
        raise ValueError(f"no material is sampled by {volume_column}")
    weights = cells[volume_column] / volume
    result = pd.DataFrame({"volume_km3": [volume * 1e-9]})
    source_depth_column = "peak_source_depth_m" if volume_column == "peak_volume_m3" else "floor_layer_source_depth_m"
    for column, output_column, factor in [
        (source_depth_column, "mean_source_depth_km", 1e-3),
        ("heated_temperature_k", "mean_heated_temperature_k", 1.0),
        ("impact_peak_pressure_pa", "mean_peak_pressure_gpa", 1e-9),
        ("thermal_reset_fraction", "mean_thermal_reset_fraction", 1.0),
        ("magnitude_a_m", "mean_local_magnitude_a_m", 1.0),
    ]:
        result[output_column] = [(cells[column] * weights).sum() * factor]
    for prefix, label in [("inherited_", "inherited"), ("trm_", "trm"), ("", "total")]:
        columns = [f"{prefix}m{axis}_a_m" for axis in "xyz"]
        mean_vector = cells[columns].multiply(weights, axis=0).sum().to_numpy()
        result[f"mean_{label}_magnitude_a_m"] = [np.linalg.norm(mean_vector)]
        for axis, value in zip("xyz", mean_vector):
            result[f"mean_{label}_m{axis}_a_m"] = [value]
            if label == "total":
                result[f"moment_{axis}_a_m2"] = [value * volume]
        if label == "total":
            result["moment_magnitude_a_m2"] = [np.linalg.norm(mean_vector) * volume]
    return result


example_dir = Path(__file__).resolve().parent
output_dir = example_dir.parent / "output" / "lunar_central_peak_depth_ratio"
output_dir.mkdir(parents=True, exist_ok=True)

# d is rim-to-floor depth and excludes the relief of the central peak
# these are synthetic geometries rather than measurements of named lunar craters
diameters_km = np.array([50.0, 75.0, 100.0])
depth_diameter_ratios = np.linspace(0.04, 0.10, 7)
terrains = ["mare", "highlands"]
radial_cells = 48
depth_cells = 96
source_depth_fraction = 0.40
peak_radius_fraction = 0.10
floor_radius_fraction = 0.30
uplift_radius_fraction = 0.20
uplift_depth_factor = 2.0
floor_layer_fraction = 0.02

# heating and pressure are prescribed in source coordinates and held fixed across depth ratios
# peak height uses a published reference fit while peak width and uplift are assumptions
temperature_rise_k = 1500.0
heating_radius_fraction = 0.18
heating_depth_fraction = 0.18
pressure_amplitude_pa = 20.0e9
pressure_radius_fraction = 0.25
pressure_depth_fraction = 0.30
initial_mx_a_m = 1.0
blocking = pd.DataFrame(
    {"blocking_k": np.linspace(400.0, 850.0, 64), "weight": np.full(64, 1 / 64)}
)

# three states suffice for the fully cooled endpoint under a constant field
# the elapsed times do not define a heat diffusion solution or a cooling rate
times_s = [0.0, 1.0, 1.0e12]
scenarios = pd.read_csv(example_dir / "planetary_scenarios.csv")
scenarios = scenarios.loc[scenarios["body"] == "moon"].copy()
if scenarios.empty or scenarios["era"].duplicated().any():
    raise ValueError("planetary_scenarios.csv requires unique lunar eras")
if not np.isfinite(diameters_km).all() or ((diameters_km < 50) | (diameters_km > 100)).any():
    raise ValueError("this central-peak experiment uses diameters from 50 to 100 km")
if diameters_km.size == 0 or depth_diameter_ratios.size == 0:
    raise ValueError("provide at least one diameter and depth-to-diameter ratio")
if not np.isfinite(depth_diameter_ratios).all() or (depth_diameter_ratios <= 0).any():
    raise ValueError("depth-to-diameter ratios must be finite and positive")

reference_cases = pd.MultiIndex.from_product(
    [diameters_km, terrains], names=["diameter_km", "terrain"]
).to_frame(index=False)
reference_cases = lunar_crater_scaling(reference_cases)
ratios = pd.DataFrame({"depth_diameter_ratio": depth_diameter_ratios})
cases = reference_cases.merge(ratios, how="cross")
cases["crater_depth_km"] = cases["diameter_km"] * cases["depth_diameter_ratio"]
cases["uplift_amplitude_km"] = uplift_depth_factor * cases["crater_depth_km"]
cases["central_source_surface_depth_km"] = (
    cases["crater_depth_km"] - cases["reference_peak_height_km"] + cases["uplift_amplitude_km"]
)
cases.to_csv(output_dir / "geometry_cases.csv", index=False)
scenarios.to_csv(output_dir / "field_scenarios.csv", index=False)
blocking.to_csv(output_dir / "blocking_spectrum.csv", index=False)
parameters = pd.DataFrame(
    [
        ["radial_cells", radial_cells, "count"],
        ["depth_cells", depth_cells, "count"],
        ["source_depth_fraction", source_depth_fraction, "fraction of diameter"],
        ["peak_radius_fraction", peak_radius_fraction, "fraction of diameter"],
        ["floor_radius_fraction", floor_radius_fraction, "fraction of diameter"],
        ["uplift_radius_fraction", uplift_radius_fraction, "fraction of diameter"],
        ["uplift_depth_factor", uplift_depth_factor, "multiple of crater depth"],
        ["floor_layer_fraction", floor_layer_fraction, "fraction of diameter"],
        ["temperature_rise_k", temperature_rise_k, "K"],
        ["heating_radius_fraction", heating_radius_fraction, "fraction of diameter"],
        ["heating_depth_fraction", heating_depth_fraction, "fraction of diameter"],
        ["pressure_amplitude_pa", pressure_amplitude_pa, "Pa"],
        ["pressure_radius_fraction", pressure_radius_fraction, "fraction of diameter"],
        ["pressure_depth_fraction", pressure_depth_fraction, "fraction of diameter"],
        ["initial_mx_a_m", initial_mx_a_m, "A/m"],
        ["fully_cooled_time_s", times_s[-1], "s"],
    ],
    columns=["parameter", "value", "units"],
)
parameters.to_csv(output_dir / "experiment_parameters.csv", index=False)

summary_frames = []
peak_frames = []
source_frames = []
section_frames = []
section_diameter_km = diameters_km[len(diameters_km) // 2]
section_ratios = np.unique(depth_diameter_ratios[[0, len(depth_diameter_ratios) // 2, -1]])
section_terrain = "highlands" if "highlands" in terrains else terrains[0]
section_era = "ancient" if "ancient" in scenarios["era"].values else scenarios["era"].iloc[0]

for diameter_km in diameters_km:
    diameter_m = diameter_km * 1000
    source_cells = axisymmetric_source_cells(
        diameter_m,
        radial_cells=radial_cells,
        depth_cells=depth_cells,
        source_depth_fraction=source_depth_fraction,
        radial_breaks=[peak_radius_fraction, uplift_radius_fraction, floor_radius_fraction],
    )
    source_cells["impact_peak_pressure_pa"] = pressure_amplitude_pa * np.exp(
        -(source_cells["radius_m"] / (pressure_radius_fraction * diameter_m)) ** 2
        - (source_cells["source_depth_m"] / (pressure_depth_fraction * diameter_m)) ** 2
    )
    heating_fraction = np.exp(
        -(source_cells["radius_m"] / (heating_radius_fraction * diameter_m)) ** 2
        - (source_cells["source_depth_m"] / (heating_depth_fraction * diameter_m)) ** 2
    )
    for scenario in scenarios.itertuples(index=False):
        source_cells["heated_temperature_k"] = scenario.ambient_temperature_k + temperature_rise_k * heating_fraction
        source_cells["thermal_reset_fraction"] = (
            (source_cells["heated_temperature_k"].to_numpy()[:, None] >= blocking["blocking_k"].to_numpy())
            * blocking["weight"].to_numpy()
        ).sum(axis=1)
        frames = []
        for time_s in times_s:
            frame = source_cells[["tracer_id"]].copy()
            frame["time_s"] = time_s
            frame["temperature_k"] = source_cells["heated_temperature_k"] if time_s == times_s[1] else scenario.ambient_temperature_k
            frame["peak_pressure_pa"] = 0.0 if time_s == times_s[0] else source_cells["impact_peak_pressure_pa"]
            frames.append(frame)
        history = pd.concat(frames, ignore_index=True)
        properties = source_cells[["tracer_id"]].copy()
        properties["initial_mx_a_m"] = initial_mx_a_m
        properties["initial_my_a_m"] = 0.0
        properties["initial_mz_a_m"] = 0.0
        properties["trm_a_m_per_t"] = scenario.trm_a_m_per_t
        field = pd.DataFrame(
            {
                "time_s": [times_s[0], times_s[-1]],
                "bx_t": scenario.background_bx_t + scenario.local_bx_t,
                "by_t": scenario.background_by_t + scenario.local_by_t,
                "bz_t": scenario.background_bz_t + scenario.local_bz_t,
            }
        )
        result = magnetize_tracers(history, properties, blocking, field, pressure_half_pa=scenario.pressure_half_pa)
        final = result.groupby("tracer_id", sort=False).tail(1)
        final = source_cells.merge(final, on="tracer_id", validate="one_to_one")
        source_frames.append(final.assign(diameter_km=diameter_km, era=scenario.era))

        for case in cases.loc[cases["diameter_km"] == diameter_km].itertuples(index=False):
            mapped = central_peak_geometry(
                final,
                diameter_m,
                case.depth_diameter_ratio,
                case.reference_peak_height_km * 1000,
                peak_radius_fraction=peak_radius_fraction,
                floor_radius_fraction=floor_radius_fraction,
                uplift_radius_fraction=uplift_radius_fraction,
                uplift_depth_factor=uplift_depth_factor,
                floor_layer_fraction=floor_layer_fraction,
            )
            mapped["diameter_km"] = diameter_km
            mapped["terrain"] = case.terrain
            mapped["depth_diameter_ratio"] = case.depth_diameter_ratio
            mapped["era"] = scenario.era
            for region, volume_column in [("central_peak", "peak_volume_m3"), ("floor_layer", "floor_layer_volume_m3")]:
                summary = summarize_region(mapped, volume_column)
                summary["diameter_km"] = diameter_km
                summary["terrain"] = case.terrain
                summary["depth_diameter_ratio"] = case.depth_diameter_ratio
                summary["crater_depth_km"] = case.crater_depth_km
                summary["peak_height_km"] = case.reference_peak_height_km
                summary["reference_depth_diameter_ratio"] = case.reference_depth_diameter_ratio
                summary["era"] = scenario.era
                summary["region"] = region
                summary_frames.append(summary)
            peak_frames.append(mapped.loc[mapped["peak_volume_m3"] > 0].copy())
            if (
                diameter_km == section_diameter_km
                and case.terrain == section_terrain
                and scenario.era == section_era
                and np.isclose(case.depth_diameter_ratio, section_ratios).any()
            ):
                section_frames.append(mapped.loc[mapped["retained_volume_m3"] > 0].copy())
    print(f"completed {diameter_km:g} km lunar crater cases", flush=True)

summary = pd.concat(summary_frames, ignore_index=True)
peak_cells = pd.concat(peak_frames, ignore_index=True)
sections = pd.concat(section_frames, ignore_index=True)
summary.to_csv(output_dir / "magnetization_summary.csv", index=False)
peak_cells.to_csv(output_dir / "central_peak_cells.csv", index=False)
pd.concat(source_frames, ignore_index=True).to_csv(output_dir / "source_magnetization.csv", index=False)
sections.to_csv(output_dir / "cross_section_cells.csv", index=False)

fig, axes = plt.subplots(2, len(diameters_km), figsize=(5 * len(diameters_km), 8), squeeze=False, layout="constrained")
colors = ["#4169a1", "#c77b3c"]
styles = ["-", "--", "-."]
for column_index, diameter_km in enumerate(diameters_km):
    for row_index, region in enumerate(["central_peak", "floor_layer"]):
        ax = axes[row_index, column_index]
        for terrain_index, terrain in enumerate(terrains):
            for era_index, era in enumerate(scenarios["era"]):
                curve = summary.loc[
                    (summary["diameter_km"] == diameter_km) & (summary["terrain"] == terrain)
                    & (summary["era"] == era) & (summary["region"] == region)
                ].sort_values("depth_diameter_ratio")
                ax.plot(
                    curve["depth_diameter_ratio"], curve["mean_total_magnitude_a_m"],
                    color=colors[terrain_index % len(colors)], linestyle=styles[era_index % len(styles)],
                    marker="o", markersize=3, label=f"{terrain.capitalize()}, {era} field",
                )
            reference_ratio = reference_cases.loc[
                (reference_cases["diameter_km"] == diameter_km) & (reference_cases["terrain"] == terrain),
                "reference_depth_diameter_ratio",
            ].iloc[0]
            ax.axvline(reference_ratio, color=colors[terrain_index % len(colors)], linestyle=":", alpha=0.6, label=f"{terrain.capitalize()} reference d/D")
        ax.set_title(f"{diameter_km:g} km crater")
        ax.set_xlabel("Rim-to-floor depth / rim-to-rim diameter, d/D")
        ax.set_ylabel(f"{'Central peak' if region == 'central_peak' else 'Floor layer'} mean |M| [A/m]")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3)
fig.suptitle("Lunar crater geometry sensitivity — prescribed heating, pressure, and uplift")
fig.savefig(output_dir / "depth_ratio_magnetization.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(len(section_ratios), 1, figsize=(10, 2.3 * len(section_ratios)), squeeze=False, layout="constrained")
norm = Normalize(vmin=0, vmax=max(abs(initial_mx_a_m), sections["magnitude_a_m"].max()))
for ax, ratio in zip(axes.ravel(), section_ratios):
    section = sections.loc[np.isclose(sections["depth_diameter_ratio"], ratio)]
    visible_top = np.maximum(section["final_top_m"], section["surface_depth_m"])
    for sign in [-1, 1]:
        vertices = np.stack(
            [
                np.column_stack((sign * section["radius_inner_m"] / 1000, -visible_top / 1000)),
                np.column_stack((sign * section["radius_outer_m"] / 1000, -visible_top / 1000)),
                np.column_stack((sign * section["radius_outer_m"] / 1000, -section["final_bottom_m"] / 1000)),
                np.column_stack((sign * section["radius_inner_m"] / 1000, -section["final_bottom_m"] / 1000)),
            ],
            axis=1,
        )
        collection = PolyCollection(vertices, array=section["magnitude_a_m"].to_numpy(), cmap="viridis", norm=norm, edgecolors="none")
        ax.add_collection(collection)
    radius = np.linspace(0, section_diameter_km / 2, 501)
    reference = reference_cases.loc[
        (reference_cases["diameter_km"] == section_diameter_km) & (reference_cases["terrain"] == section_terrain)
    ].iloc[0]
    crater_depth = ratio * section_diameter_km
    bowl_depth = crater_depth * np.clip(
        (section_diameter_km / 2 - radius) / (section_diameter_km / 2 - floor_radius_fraction * section_diameter_km), 0, 1
    )
    peak_relief = reference["reference_peak_height_km"] * np.clip(1 - (radius / (peak_radius_fraction * section_diameter_km)) ** 2, 0, 1)
    ax.plot(np.concatenate((-radius[:0:-1], radius)), -np.concatenate((bowl_depth[:0:-1] - peak_relief[:0:-1], bowl_depth - peak_relief)), color="black", linewidth=1)
    ax.axhline(-crater_depth, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlim(-section_diameter_km / 2, section_diameter_km / 2)
    ax.set_ylim(-section_diameter_km * (depth_diameter_ratios.max() + 0.06), 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"d/D = {ratio:.3f}")
    ax.set_xlabel("Distance from crater center [km]")
    ax.set_ylabel("Elevation relative to rim [km]")
fig.colorbar(collection, ax=axes.ravel().tolist(), label="Final magnetization magnitude [A/m]", shrink=0.7)
fig.suptitle(f"{section_diameter_km:g} km {section_terrain} crater, {section_era} field — synthetic central peaks")
fig.savefig(output_dir / "central_peak_cross_sections.png", dpi=180)
plt.close(fig)

print(f"modeled {len(cases) * len(scenarios)} geometry/field combinations")
print(f"outputs saved to {output_dir}")
