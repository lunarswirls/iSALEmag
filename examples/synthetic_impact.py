"""Run an illustrative impact-heating and cooling scenario"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isalemag import magnetize_tracers


# these parameters illustrate behavior and are not calibrated to a specific rock
output_dir = Path(__file__).resolve().parents[1] / "output" / "synthetic_impact"
output_dir.mkdir(parents=True, exist_ok=True)
ambient_k = 293.0
cooling_scale_s = 3.0e8
pressure_half_pa = 3.0e9
field_strength_t = 50.0e-6
trm_a_m_per_t = 2.0e4
initial_mx_a_m = 1.0
blocking = pd.DataFrame(
    {"blocking_k": np.linspace(400.0, 850.0, 64), "weight": np.full(64, 1.0 / 64)}
)

# prescribe heating and single-event peak pressure on a fixed r-z grid
radius_m, depth_m = np.meshgrid(np.linspace(0, 20000, 41), np.linspace(0, 10000, 21))
tracers = pd.DataFrame(
    {
        "tracer_id": np.arange(radius_m.size),
        "radius_m": radius_m.ravel(),
        "depth_m": depth_m.ravel(),
    }
)
distance_squared = tracers["radius_m"] ** 2 + (tracers["depth_m"] + 1500.0) ** 2
tracers["heated_temperature_k"] = ambient_k + 1500 * np.exp(-distance_squared / 7000.0**2)
tracers["impact_peak_pressure_pa"] = 20.0e9 * np.exp(-distance_squared / 8500.0**2)
times_s = np.concatenate(([0.0, 1.0], np.geomspace(1.0e5, 3.0e9, 48)))
frames = []
for time_s in times_s:
    frame = tracers[["tracer_id", "radius_m", "depth_m"]].copy()
    frame["time_s"] = time_s
    if time_s == 0.0:
        frame["temperature_k"] = ambient_k
        frame["peak_pressure_pa"] = 0.0
    else:
        # exponential cooling is a prescribed history, not a heat diffusion calculation
        frame["temperature_k"] = ambient_k + (
            tracers["heated_temperature_k"] - ambient_k
        ) * np.exp(-(time_s - 1.0) / cooling_scale_s)
        frame["peak_pressure_pa"] = tracers["impact_peak_pressure_pa"]
    frames.append(frame)
history = pd.concat(frames, ignore_index=True)

properties = tracers[["tracer_id"]].copy()
properties["initial_mx_a_m"] = initial_mx_a_m
properties["initial_my_a_m"] = 0.0
properties["initial_mz_a_m"] = 0.0
properties["trm_a_m_per_t"] = trm_a_m_per_t
field = pd.DataFrame(
    {"time_s": [times_s[0], times_s[-1]], "bx_t": 0.0, "by_t": 0.0, "bz_t": field_strength_t}
)
result = magnetize_tracers(
    history, properties, blocking, field, pressure_half_pa=pressure_half_pa
)
final = result.groupby("tracer_id", sort=False).tail(1).copy()
final = final.merge(
    tracers[["tracer_id", "heated_temperature_k", "impact_peak_pressure_pa"]],
    on="tracer_id",
    validate="one_to_one",
)
result.to_csv(output_dir / "magnetization_history.csv", index=False)
final.to_csv(output_dir / "final_magnetization.csv", index=False)
blocking.to_csv(output_dir / "blocking_spectrum.csv", index=False)
properties.to_csv(output_dir / "tracer_properties.csv", index=False)
field.to_csv(output_dir / "ambient_field.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
panels = [
    ("heated_temperature_k", "Prescribed heating", "Temperature [K]"),
    ("impact_peak_pressure_pa", "Prescribed shock pressure", "Pressure [GPa]"),
    ("inherited_mx_a_m", "Surviving initial remanence", "Magnetization [A/m]"),
    ("trm_mz_a_m", "Remanence acquired during cooling", "Magnetization [A/m]"),
]
for ax, (column, title, label) in zip(axes.ravel(), panels):
    values = final[column].to_numpy().reshape(radius_m.shape)
    if column == "impact_peak_pressure_pa":
        values = values * 1e-9
    panel = ax.pcolormesh(radius_m / 1000, depth_m / 1000, values, shading="auto", cmap="viridis")
    if column.endswith("a_m"):
        panel.set_clim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Radius [km]")
    ax.set_ylabel("Depth [km]")
    ax.set_aspect("equal")
    fig.colorbar(panel, ax=ax, label=label)
axes[0, 0].invert_yaxis()
fig.suptitle("Synthetic impact magnetization — illustrative parameters")
fig.savefig(output_dir / "magnetization.png", dpi=180)
plt.close(fig)

print(f"modeled {len(tracers)} tracers at {len(times_s)} times")
print(f"outputs saved to {output_dir}")
