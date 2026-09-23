# PyDiffChron

PyDiffChron contains the numerical calculations and figure-generation workflows developed for the manuscript:

*“Hold on: assessing uncertainty and hidden time in diffusion chronometry”* by Joaquín Bastías-Silva, Adam J. R. Kent, and Luca Caricchi, submitted to the Journal of Petrology.

## Overview

Diffusion chronometry is widely used to estimate the timescales of magmatic processes, but the resulting timescales depend strongly on the thermal conditions assumed during diffusion.

PyDiffChron explores the temperature–time non-uniqueness inherent in diffusion chronometry using synthetic Ba and Sr concentration profiles in plagioclase. The calculations investigate:

- non-uniqueness under isothermal and non-isothermal conditions;
- the influence of analytical uncertainty on admissible diffusion solutions;
- hidden residence time that may remain unresolved within analytical uncertainty;
- the dependence of hidden time on temperature, plagioclase composition, and diffusion duration;
- the use of paired Sr–Ba diffusion widths to constrain effective diffusion temperature;
- recovery of corresponding diffusion timescales without fixing temperature or time a priori.

## Main script

The numerical calculations are contained in:

```text
PyDiffChron_main.py
```
The full script should be run from top to bottom.

## Installation

The required Python packages are listed in `requirements.txt`.

The main external dependencies are:

- NumPy
- Matplotlib
- SciPy
- tqdm

## Running the calculations

The script creates a local directory named:

```text
Figures/
```

and saves the generated figures there.

Some calculations involve large parameter-space searches and may therefore require several minutes to complete, depending on the equipment used.

## Analytical uncertainty

Several figure blocks include a switch controlling which analytical uncertainty model is used for model acceptance.

For example:

```python
use_EPMA_fig3 = False
```

where:

- `False` uses the **SIMS** uncertainty envelope;
- `True` uses the **EPMA** uncertainty envelope.

The version distributed with the manuscript is currently configured to use the SIMS uncertainty envelope for the relevant calculations.

The analytical uncertainty assumptions used in the script are:

- SIMS: ±max(5%, 5 ppm)
- LA-ICP-MS: ±max(8%, 8 ppm)
- EPMA: ±max(15%, 15 ppm)

## Diffusion parameterisation

The default Sr and Ba diffusion parameterisations are based on **Grocolas et al. (2025)**.

Alternative diffusion calibrations can be implemented by modifying:

```python
diffusion_log10D_Sr()
diffusion_log10D_Ba()
```

## Output and reproducibility

The calculations generate the synthetic profiles, uncertainty envelopes, accepted model populations, hidden-residence-time calculations, and paired-element diffusion-width analyses used to construct the manuscript figures.

Random subsampling used for visualisation in selected figures uses a fixed random seed so that the displayed model populations are reproducible. Subsampling affects only the plotted profiles and does not alter model acceptance or the calculated statistics.

## Authors

- Joaquín Bastías-Silva
- Adam J. R. Kent
- Luca Caricchi

## Citation

A permanent citation and DOI will be added following archival of the version 1.0.0 release on Zenodo.

Until then, please cite the associated manuscript:

> Bastías-Silva, J., Kent, A. J. R., and Caricchi, L.  
> *Hold on: assessing uncertainty and hidden time in diffusion chronometry*.  
> Submitted to the *Journal of Petrology*.

## Licence

The source code is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
