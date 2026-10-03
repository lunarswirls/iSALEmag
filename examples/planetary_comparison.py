"""Compare assumed ancient and present background fields for three bodies"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isalemag import magnetize_tracers


example_dir = Path(__file__).resolve().parent
output_dir = example_dir.parent / "output" / "planetary_comparison"
output_dir.mkdir(parents=True, exist_ok=True)
scenarios = pd.read_csv(example_dir / "planetary_scenarios.csv")

# hold thermal history and carrier properties fixed to isolate the field comparison
# ancient fields and material parameters are assumptions rather than reconstructions
times_s = np.concatenate(([0.0, 1.0], np.geomspace(1.0e5, 3.0e9, 90)))
cooling_scale_s = 3.0e8
blocking = pd.DataFrame(
    {"blocking_k": np.linspace(400.0, 850.0, 64), "weight": np.full(64, 1.0 / 64)}
)
tracers = pd.DataFrame(
    {
        "tracer_id": [0, 1, 2],
        "thermal_case": ["Below blocking", "Partial reset", "Full reset"],
        "heated_fraction": [0.03, 0.34, 1.0],
        "impact_peak_pressure_pa": [0.0, 3e9, 10e9],
    }
)
histories = []
for scenario in scenarios.itertuples(index=False):
    frames = []
    for time_s in times_s:
        frame = tracers[["tracer_id", "thermal_case"]].copy()
        frame["body"] = scenario.body
        frame["era"] = scenario.era
        frame["time_s"] = time_s
        if time_s == 0.0:
            frame["temperature_k"] = scenario.ambient_temperature_k
            frame["peak_pressure_pa"] = 0.0
        else:
            frame["temperature_k"] = scenario.ambient_temperature_k + (
                scenario.heated_temperature_k - scenario.ambient_temperature_k
            ) * tracers["heated_fraction"] * np.exp(-(time_s - 1) / cooling_scale_s)
            frame["peak_pressure_pa"] = tracers["impact_peak_pressure_pa"]
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    properties = tracers[["tracer_id"]].copy()
    properties["initial_mx_a_m"] = 1.0
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
    result = magnetize_tracers(
        history, properties, blocking, field, pressure_half_pa=scenario.pressure_half_pa
    )
    histories.append(result)

result = pd.concat(histories, ignore_index=True)
final = result.groupby(["body", "era", "tracer_id"], sort=False).tail(1).copy()
final["inherited_magnitude_a_m"] = np.linalg.norm(
    final[["inherited_mx_a_m", "inherited_my_a_m", "inherited_mz_a_m"]], axis=1
)
final["trm_magnitude_a_m"] = np.linalg.norm(
    final[["trm_mx_a_m", "trm_my_a_m", "trm_mz_a_m"]], axis=1
)
result.to_csv(output_dir / "magnetization_history.csv", index=False)
final.to_csv(output_dir / "final_magnetization.csv", index=False)
scenarios.to_csv(output_dir / "scenario_parameters.csv", index=False)
blocking.to_csv(output_dir / "blocking_spectrum.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout="constrained")
positions = np.arange(len(tracers))
for ax, body in zip(axes, scenarios["body"].drop_duplicates()):
    for era, offset, color in [("ancient", -0.18, "#4169a1"), ("present", 0.18, "#ce8153")]:
        case = final[(final["body"] == body) & (final["era"] == era)].sort_values("tracer_id")
        bars = ax.bar(
            positions + offset,
            case["magnitude_a_m"],
            width=0.34,
            color=color,
            label="Assumed ancient field" if era == "ancient" else "Present reference field",
        )
        ax.bar_label(bars, labels=[f"{value:.3g}" for value in case["magnitude_a_m"]], padding=3)
    ax.set_title(body.capitalize())
    ax.set_xticks(positions, tracers["thermal_case"], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Final total remanence [A/m]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2)
fig.suptitle("Field comparison with shared illustrative thermal and magnetic properties")
fig.savefig(output_dir / "planetary_comparison.png", dpi=180)
plt.close(fig)

columns = ["body", "era", "thermal_case", "inherited_magnitude_a_m", "trm_magnitude_a_m", "magnitude_a_m"]
print(final[columns].to_string(index=False))
print(f"outputs saved to {output_dir}")
