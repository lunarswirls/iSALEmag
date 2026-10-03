# iSALEmag

Model surviving impact remanence and thermoremanent magnetization (TRM) acquired during cooling, using prescribed temperature and pressure histories for individual tracers. Inputs and outputs are pandas dataframes; all magnetic vectors use one fixed Cartesian coordinate frame.

This is a starting model for impact-heated rocks on the Moon, Mars, and Mercury. It post-processes histories and does not calculate impact dynamics or heat diffusion. The example parameters are illustrative, not a calibration for a planet or material.

## Run the example

The existing `gravmagpy` environment has the required numerical and plotting packages:

```bash
cd /Users/danywaller/code/iSALEmag
PYTHONPATH=src MPLCONFIGDIR=/tmp/isalemag-mpl MPLBACKEND=Agg /Users/danywaller/code/venvs/gravmagpy/bin/python examples/synthetic_impact.py
```

The example writes magnetic histories, final tracer magnetization, input magnetic properties, and a cross-section figure under `output/synthetic_impact/`. It prescribes a fixed spatial heating pattern followed by exponential cooling. It does not simulate crater formation, tracer motion, or conductive cooling.

For use outside this checkout, install the package with the chosen environment's Python: `python -m pip install -e '.[plot]'`.

## Ancient and present field comparison

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/isalemag-mpl MPLBACKEND=Agg /Users/danywaller/code/venvs/gravmagpy/bin/python examples/planetary_comparison.py
```

Edit `examples/planetary_scenarios.csv` to set each body's ancient and present fields, local fields, reference temperature, TRM coefficient, and pressure scale. The script compares material below its blocking range, partially reset material, and fully reset material for all six body/era combinations. It saves the input scenarios, complete histories, final component magnitudes, and a comparison figure under `output/planetary_comparison/`.

The ancient cases all assume a constant 50 µT paleofield as a controlled comparison; these are not estimates of the historical fields. The present Moon and Mars cases set the global background to zero, consistent with their lack of an active global dynamo field ([Moon](https://science.nasa.gov/moon/solar-wind/), [Mars](https://science.nasa.gov/mars/facts/)). Their local crustal and external fields can still be nonzero: enter a prescribed local vector in the `local_b*_t` columns. A zero local field is a selected baseline, not a claim that the surface field vanishes everywhere.

The present Mercury case uses a constant 200 nT reference vector as a weak-field scenario. This amplitude reflects the scale of its intrinsic dipole, whose measured moment is about 190 nT times the cube of Mercury's radius ([Anderson et al., 2012](https://doi.org/10.1029/2012JE004159)). It is not a field prediction for a specific site; Mercury's dipole offset and latitude dependence require a spatial field model or a supplied local history. The comparison holds the thermal history, blocking spectrum, and initial remanence fixed across bodies to isolate field effects. Actual lithology, surface/subsurface temperature, and cooling conditions need separate inputs for each impact. The nominal 293 K reference is shared for this comparison and is not a planet-wide surface temperature.

## Inputs and use

| Dataframe | Required columns | Meaning |
|---|---|---|
| `history` | `tracer_id`, `time_s`, `temperature_k` | One sample per tracer and time; arbitrary additional columns are preserved |
| `properties` | `tracer_id`, `initial_mx_a_m`, `initial_my_a_m`, `initial_mz_a_m`, `trm_a_m_per_t` | One row per tracer; initial remanence in A/m and a calibrated TRM coefficient in A/m/T |
| `blocking` | `blocking_k`, `weight` | A shared discrete blocking spectrum; nonnegative weights must sum to one |
| `field` | `time_s`, `bx_t`, `by_t`, `bz_t` | Ambient field in tesla; samples must cover all history times |

Enable optional pressure demagnetization with a positive `pressure_half_pa`. In that case `history` must also contain `peak_pressure_pa` or `pressure_pa`, in pascals. The peak-pressure column takes precedence. Tensile instantaneous pressure does not demagnetize material in this pressure prescription.

```python
import pandas as pd
from isalemag import magnetize_tracers

history = pd.read_csv("tracer_history.csv")
properties = pd.read_csv("tracer_properties.csv")
blocking = pd.read_csv("blocking_spectrum.csv")
field = pd.read_csv("ambient_field.csv")

result = magnetize_tracers(history, properties, blocking, field)
final = result.groupby("tracer_id", sort=False).tail(1)
final.to_csv("final_magnetization.csv", index=False)
```

Outputs include `mx_a_m`, `my_a_m`, `mz_a_m`, `magnitude_a_m`, the separate `inherited_m*_a_m` and `trm_m*_a_m` components, and `blocked_fraction`. Pressure runs also report `modeled_peak_pressure_pa` and `pressure_retention`. The retention column describes the empirical pressure law alone; it is not the fraction of total magnetization remaining after cooling. Magnetization magnitude is the norm of the vector sum, so opposing components can cancel.

## Model assumptions

The initial remanence and the full TRM capacity are distributed across the same supplied blocking weights. Heating to or above a bin's temperature erases that bin. Cooling below it records

```text
trm_vector = weight * trm_a_m_per_t * ambient_field_vector_t
```

The ambient field is interpolated at the crossing time, with temperature assumed linear between history samples. A bin that stays cold preserves its inherited component and gains no new TRM. Reheating erases any acquired TRM before subsequent cooling replaces it. Hot material receives no future cooling remanence until that cooling appears in the input history.

Optional shock loss follows the adjustable empirical law

```text
retention = exp(-ln(2) * (peak_pressure_pa / pressure_half_pa)**pressure_exponent)
```

Thus `pressure_half_pa` is the 50% retention pressure. Each increase in a tracer's running maximum pressure attenuates remanence present at that sample by the corresponding retention ratio. Repeated output of the same peak does not accumulate loss. TRM acquired later is not attenuated by an earlier shock. This is a user-selected phenomenological law, not a published universal demagnetization relation, and it describes one impact shock. Pressure attenuation precedes thermal processing within each sample.

Choose blocking temperatures from measured magnetic spectra and restrict them to the carrier's physical temperature range. The coefficient and pressure parameters need material-specific calibration. The model equates blocking and unblocking temperatures, treats thermal unblocking as instantaneous, and assumes linear TRM acquisition in a weak field. It omits relaxation kinetics, cooling-rate shifts of blocking temperatures, induced magnetization, shock remanence acquisition, chemical changes, magnetic interactions, and rotation of rocks or ejecta. A brief shock-temperature spike can therefore be overinterpreted as thermal resetting. Resolve heating and cooling in time and compare results with refined histories and blocking spectra before using them quantitatively.

## iSALE histories

The adapter accepts an already opened `pySALEPlot` model, leaving its lifetime and scaling under the caller's control:

```python
import pySALEPlot as psp
from isalemag import read_isale_tracers

model = psp.opendatfile("jdata.dat")
history = read_isale_tracers(model)
history.to_csv("tracer_history.csv", index=False)
```

Save instantaneous tracer temperature `Trt` and cumulative peak pressure `TrP` in the iSALE `TR_VAR` setting, for example `#Trt-TrP-TrM#`. `TrT` is peak temperature and cannot represent cooling. The adapter returns positions as `x_native`, `y_native`, and, for 3D models, `z_native`; they retain the caller-selected distance scale. In an axisymmetric 2D calculation, these are radius and vertical position, not a Cartesian magnetic-vector basis. Magnetic directions must be supplied consistently in the chosen fixed frame.

Hydrocode histories often end before geological cooling completes. Extend them with a suitable thermal calculation if the goal is final cooled remanence. Supply the first history sample before heating when possible. No numerical `jdata.dat` run has been validated here: `pySALEPlot` is not installed in the selected environment, and its adapter is checked against the neighboring source API and controlled test inputs.

## Verification and references

```bash
PYTHONPATH=src /Users/danywaller/code/venvs/gravmagpy/bin/python -m unittest discover -s tests -v
```

Checks cover full and partial thermal resetting, zero-field cooling, field changes during blocking, reheating, pressure retention, vector cancellation, independent tracers, input validation, and adapter field selection.

The separation of shock modification and thermal remanence is motivated by [Gattacceca et al. (2010)](https://doi.org/10.1016/j.pepi.2010.06.009). Laboratory evidence that shock acquisition depends on both pressure and temperature is described by [Sato et al. (2024)](https://doi.org/10.1029/2023JE007864); that acquisition mechanism is not implemented in this starter model. The limits of fixed blocking thresholds are discussed by [Berndt et al. (2021)](https://doi.org/10.1093/gji/ggaa514).
