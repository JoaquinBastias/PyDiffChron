"""
PyDiffChron
===========

This code accompanies the manuscript:
"Hold on: assessing uncertainty and hidden time in diffusion chronometry"
by Joaquín Bastías-Silva, Adam J. R. Kent, and Luca Caricchi, submitted to
the Journal of Petrology.

The code in this script contains the numerical calculations and figure-generation
workflows associated with the manuscript. The calculations are organised according
to their computational dependencies, with shared functions, model definitions, and
intermediate results. 

Please note that the first two figures do not follow the numerical order in which 
they appear in the manuscript.

The script is intended to provide a transparent and reproducible record of the
numerical experiments used in the manuscript, including the exploration of
temperature–time non-uniqueness, analytical uncertainty, hidden residence time,
and multi-element constraints in diffusion chronometry.
"""













#region MANUSCRIPT FIGURE 2 — DIFFUSION PROFILE PROPAGATION
# ============================================================
# MANUSCRIPT FIGURE 2 — DIFFUSION PROFILE PROPAGATION
# ============================================================

# Manuscript Figure 2 compares Sr and Ba diffusion profiles in plagioclase under
# isothermal and non-isothermal conditions. Numerical finite-difference
# solutions are compared with analytical error-function solutions and
# integrated-diffusivity solutions for time-dependent temperature histories.
#
# The calculations below also define the core diffusion functions and model
# parameters that are reused in subsequent figures.

# ============================================================
# 1. IMPORTS
# ============================================================

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2. OUTPUT
# ============================================================

# Directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Figures are saved in a local "Figures" folder beside the script
OUTPUT_DIR = SCRIPT_DIR / "Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. MODEL PARAMETERS
# ============================================================

# Step concentrations (ppm)
Sr_high, Sr_low = 900.0, 400.0
Ba_high, Ba_low = 300.0, 100.0

# Plagioclase composition
X_An = 0.50

# Isothermal reference temperature (°C)
T_C = 1000.0

# Diffusion duration
t_years = 300.0
seconds_per_year = 365.25 * 24.0 * 3600.0
t_s = t_years * seconds_per_year

# Spatial grid (µm)
x0_um = 0.0
x_um = np.linspace(-300.0, 300.0, 1301)

# Constant XAn profile.
# This can later be replaced by a spatially variable composition.
X_An_profile = np.full_like(
    x_um,
    X_An,
    dtype=float
)


# ============================================================
# 4. DIFFUSION PARAMETERISATION
# ============================================================

R = 8.314462618  # Gas constant, J mol^-1 K^-1

# Default parameterisation: Grocolas et al. (2025, Earth and Planetary Science Letters)
# To use another diffusion calibration, modify the two functions below. 
def diffusion_log10D_Sr(T_K, X_An):
    """
    Return log10 Sr diffusivity in plagioclase.

    Parameters
    ----------
    T_K : float or array-like
        Temperature in Kelvin.
    X_An : float or array-like
        Anorthite mole fraction.

    Returns
    -------
    float or ndarray
        log10 diffusion coefficient, with D in m2/s.
    """

    return (
        (-1.65 * X_An)
        - 3.03
        - (368_142.0 / (2.303 * R * T_K))
    )


def diffusion_log10D_Ba(T_K, X_An):
    """
    Return log10 Ba diffusivity in plagioclase.

    Parameters
    ----------
    T_K : float or array-like
        Temperature in Kelvin.
    X_An : float or array-like
        Anorthite mole fraction.

    Returns
    -------
    float or ndarray
        log10 diffusion coefficient, with D in m2/s.
    """

    return (
        (-1.43 * X_An)
        - 4.65
        - (337_037.0 / (2.303 * R * T_K))
    )


def diffusion_coefficient(T_K, X_An, species):
    """
    Return diffusion coefficient D in m2/s.

    """

    species = species.lower()

    if species == "sr":
        return 10 ** diffusion_log10D_Sr(T_K, X_An)

    if species == "ba":
        return 10 ** diffusion_log10D_Ba(T_K, X_An)

    raise ValueError("species must be 'Sr' or 'Ba'")


# ============================================================
# 5. DIFFUSION FUNCTIONS
# ============================================================

def step_initial_profile(
    x_um,
    C_high,
    C_low,
    x0_um=0.0
):
    """
    Generate an initial sharp compositional step.
    """

    return np.where(
        x_um < x0_um,
        C_high,
        C_low
    )


def erf_profile_from_integral(
    x_um,
    I_m2,
    C_high,
    C_low,
    x0_um=0.0
):
    """
    Analytical error-function profile using integrated diffusivity.

    Parameters
    ----------
    I_m2 : float
        Integrated diffusivity:

            I = integral D(T) dt

        with units of m2.
    """

    x_m = (x_um - x0_um) * 1e-6

    z = x_m / (
        2.0 * np.sqrt(I_m2)
    )

    return (
        C_low
        + 0.5
        * (C_high - C_low)
        * (1.0 - np.vectorize(math.erf)(z))
    )


def diffuse_1D_variable_XAn(
    T_series_K,
    t_total_s,
    x_um,
    C_init,
    X_An_profile,
    species
):
    """
    Solve one-dimensional diffusion using an explicit
    finite-difference scheme:

        dC/dt = d/dx [D(x,t) dC/dx]

    Diffusivity may vary with both temperature and plagioclase
    composition:

        D = D(T(t), XAn(x))

    No-flux boundary conditions are imposed at both ends.
    """

    x_m = x_um * 1e-6
    dx = x_m[1] - x_m[0]

    n_steps = len(T_series_K)
    dt = t_total_s / (n_steps - 1)

    C = C_init.copy().astype(float)

    for n in range(n_steps - 1):

        T_K = T_series_K[n]

        # Diffusivity at grid nodes
        D_x = diffusion_coefficient(
            T_K,
            X_An_profile,
            species
        )

        # Explicit finite-difference stability criterion
        alpha_max = np.max(D_x) * dt / dx**2

        if alpha_max > 0.45:
            raise RuntimeError(
                f"UNSTABLE: alpha_max={alpha_max:.3f}. "
                "Increase the number of time steps."
            )

        # Diffusivity at cell interfaces
        D_half = 0.5 * (
            D_x[:-1] + D_x[1:]
        )

        C_new = C.copy()

        # Conservative flux-form update
        C_new[1:-1] = (
            C[1:-1]
            + (dt / dx**2)
            * (
                D_half[1:]
                * (C[2:] - C[1:-1])
                -
                D_half[:-1]
                * (C[1:-1] - C[:-2])
            )
        )

        # No-flux boundaries
        C_new[0] = C_new[1]
        C_new[-1] = C_new[-2]

        C = C_new

    return C


# ============================================================
# 6. INITIAL PROFILES
# ============================================================

Sr_initial = step_initial_profile(
    x_um,
    Sr_high,
    Sr_low,
    x0_um
)

Ba_initial = step_initial_profile(
    x_um,
    Ba_high,
    Ba_low,
    x0_um
)


# ============================================================
# 7. ISOTHERMAL MODEL
# ============================================================

T_K = T_C + 273.15

D_Sr = diffusion_coefficient(
    T_K,
    X_An,
    "Sr"
)

D_Ba = diffusion_coefficient(
    T_K,
    X_An,
    "Ba"
)


# ------------------------------------------------------------
# Stable finite-difference timestep
# ------------------------------------------------------------

x_m = x_um * 1e-6
dx = x_m[1] - x_m[0]

D_Sr_x = diffusion_coefficient(
    T_K,
    X_An_profile,
    "Sr"
)

D_Ba_x = diffusion_coefficient(
    T_K,
    X_An_profile,
    "Ba"
)

D_max = max(
    np.max(D_Sr_x),
    np.max(D_Ba_x)
)

dt_max = 0.4 * dx**2 / D_max

n_steps_fd = int(
    np.ceil(t_s / dt_max)
)


T_series_const = np.full(
    n_steps_fd,
    T_K
)


Sr_profile = diffuse_1D_variable_XAn(
    T_series_const,
    t_s,
    x_um,
    Sr_initial,
    X_An_profile,
    "Sr"
)

Ba_profile = diffuse_1D_variable_XAn(
    T_series_const,
    t_s,
    x_um,
    Ba_initial,
    X_An_profile,
    "Ba"
)


# ============================================================
# 8. NON-ISOTHERMAL MODEL
# ============================================================

# Linear thermal history
Tinitial_Model = 1000.0
Tfinal_Model = 900.0

# Number of points used for integrated-D calculations
n_steps_integral = 400


if Tinitial_Model > Tfinal_Model:
    thermal_case = "cooling"

elif Tinitial_Model < Tfinal_Model:
    thermal_case = "heating"

else:
    thermal_case = "isothermal"


# ------------------------------------------------------------
# Determine stable finite-difference timestep
# ------------------------------------------------------------

t_series_tmp = np.linspace(
    0.0,
    t_s,
    500
)

T_series_C_tmp = (
    Tinitial_Model
    + (Tfinal_Model - Tinitial_Model)
    * (t_series_tmp / t_s)
)

T_series_K_tmp = (
    T_series_C_tmp + 273.15
)

# Maximum temperature corresponds to maximum diffusivity
T_max_K = np.max(
    T_series_K_tmp
)

D_Sr_max_x = diffusion_coefficient(
    T_max_K,
    X_An_profile,
    "Sr"
)

D_Ba_max_x = diffusion_coefficient(
    T_max_K,
    X_An_profile,
    "Ba"
)

D_max = max(
    np.max(D_Sr_max_x),
    np.max(D_Ba_max_x)
)

dt_max = 0.4 * dx**2 / D_max

n_steps_fd = int(
    np.ceil(t_s / dt_max)
)



# ------------------------------------------------------------
# Thermal history
# ------------------------------------------------------------

t_series = np.linspace(
    0.0,
    t_s,
    n_steps_fd
)

T_series_C = (
    Tinitial_Model
    + (Tfinal_Model - Tinitial_Model)
    * (t_series / t_s)
)

T_series_K = (
    T_series_C + 273.15
)


Sr_profile_variableT = diffuse_1D_variable_XAn(
    T_series_K,
    t_s,
    x_um,
    Sr_initial,
    X_An_profile,
    "Sr"
)

Ba_profile_variableT = diffuse_1D_variable_XAn(
    T_series_K,
    t_s,
    x_um,
    Ba_initial,
    X_An_profile,
    "Ba"
)


# ============================================================
# 9. ANALYTICAL / INTEGRATED-DIFFUSIVITY SOLUTIONS
# ============================================================

# ------------------------------------------------------------
# Isothermal
# ------------------------------------------------------------

I_Sr_constT = D_Sr * t_s
I_Ba_constT = D_Ba * t_s


Sr_profile_erf_constT = erf_profile_from_integral(
    x_um,
    I_Sr_constT,
    Sr_high,
    Sr_low,
    x0_um
)

Ba_profile_erf_constT = erf_profile_from_integral(
    x_um,
    I_Ba_constT,
    Ba_high,
    Ba_low,
    x0_um
)


# ------------------------------------------------------------
# Non-isothermal
# ------------------------------------------------------------

# Use the same temperature history calculated for the finite-difference solution.
t_series_erf = np.linspace(
    0.0,
    t_s,
    len(T_series_K)
)

D_Sr_series = diffusion_coefficient(
    T_series_K,
    X_An,
    "Sr"
)

D_Ba_series = diffusion_coefficient(
    T_series_K,
    X_An,
    "Ba"
)

I_Sr_varT = np.trapz(
    D_Sr_series,
    t_series_erf
)

I_Ba_varT = np.trapz(
    D_Ba_series,
    t_series_erf
)


Sr_profile_erf_variableT = erf_profile_from_integral(
    x_um,
    I_Sr_varT,
    Sr_high,
    Sr_low,
    x0_um
)

Ba_profile_erf_variableT = erf_profile_from_integral(
    x_um,
    I_Ba_varT,
    Ba_high,
    Ba_low,
    x0_um
)


# ============================================================
# 10. FIGURE 2
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7.5),
    sharex=True
)


# ------------------------------------------------------------
# Panel a — Sr
# ------------------------------------------------------------

axes[0].plot(
    x_um,
    Sr_initial,
    color="black",
    linewidth=2.5,
    label="Initial (t=0)"
)

axes[0].plot(
    x_um,
    Sr_profile,
    color="blue",
    linewidth=2.5,
    label=(
        f"FD derived "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )
)

axes[0].plot(
    x_um,
    Sr_profile_variableT,
    color="blue",
    linestyle="--",
    linewidth=2.5,
    label=(
        f"FD derived "
        f"({t_years:.0f} yr; "
        f"T {Tinitial_Model:.0f}→{Tfinal_Model:.0f} °C)"
    )
)

axes[0].plot(
    x_um,
    Sr_profile_erf_constT,
    color="cyan",
    linewidth=7.0,
    zorder=1,
    alpha=0.7,
    label=(
        f"Analytical erf "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )
)

axes[0].plot(
    x_um,
    Sr_profile_erf_variableT,
    color="cyan",
    linestyle="--",
    linewidth=7.0,
    zorder=1,
    alpha=0.7,
    label=(
        f"Erf-like, integrated D(T) "
        f"({t_years:.0f} yr; "
        f"T {Tinitial_Model:.0f}→{Tfinal_Model:.0f} °C)"
    )
)

axes[0].set_ylabel(
    "Sr (ppm)"
)

axes[0].set_title(
    f"Sr diffusion profile "
    f"({Sr_high:.0f} → {Sr_low:.0f} ppm)"
)

axes[0].text(
    0.98,
    0.98,
    "a",
    transform=axes[0].transAxes,
    ha="right",
    va="top",
    fontsize=22,
    fontweight="bold"
)

axes[0].legend()


# ------------------------------------------------------------
# Panel b — Ba
# ------------------------------------------------------------

axes[1].plot(
    x_um,
    Ba_initial,
    color="black",
    linewidth=2.5,
    label="Initial (t=0)"
)

axes[1].plot(
    x_um,
    Ba_profile,
    color="red",
    linewidth=2.5,
    label=(
        f"FD derived "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )
)

axes[1].plot(
    x_um,
    Ba_profile_variableT,
    color="red",
    linestyle="--",
    linewidth=2.5,
    label=(
        f"FD derived "
        f"({t_years:.0f} yr; "
        f"T {Tinitial_Model:.0f}→{Tfinal_Model:.0f} °C)"
    )
)

axes[1].plot(
    x_um,
    Ba_profile_erf_constT,
    color="orange",
    linewidth=7.0,
    zorder=1,
    alpha=0.7,
    label=(
        f"Analytical erf "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )
)

axes[1].plot(
    x_um,
    Ba_profile_erf_variableT,
    color="orange",
    linestyle="--",
    linewidth=7.0,
    zorder=1,
    alpha=0.7,
    label=(
        f"Erf-like, integrated D(T) "
        f"({t_years:.0f} yr; "
        f"T {Tinitial_Model:.0f}→{Tfinal_Model:.0f} °C)"
    )
)

axes[1].set_xlabel(
    "Distance (µm)"
)

axes[1].set_ylabel(
    "Ba (ppm)"
)

axes[1].set_title(
    f"Ba diffusion profile "
    f"({Ba_high:.0f} → {Ba_low:.0f} ppm)"
)

axes[1].text(
    0.98,
    0.98,
    "b",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=22,
    fontweight="bold"
)

axes[1].legend()


# ============================================================
# 11. SAVE FIGURE
# ============================================================

plt.tight_layout()

output_file = (
    OUTPUT_DIR
    / "Figure_2_Plagioclase_Sr_Ba_diffusion_model_propagations.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# END OF MANUSCRIPT FIGURE 2
# ============================================================
#endregion








#region MANUSCRIPT FIGURE 1 — ANALYTICAL UNCERTAINTY ENVELOPES

# ============================================================
# MANUSCRIPT FIGURE 1 — ANALYTICAL UNCERTAINTY ENVELOPES
# ============================================================

# Manuscript Figure 1 illustrates how representative analytical 
# uncertainties affect the Ba diffusion profile calculated under 
# isothermal conditions.
#
# The reference profile is inherited directly from Manuscript Figure 2:
# Ba_profile_erf_constT.
#
# No new diffusion calculation is performed in this section.
# Instead, analytical uncertainty envelopes are constructed around
# the existing Ba profile using representative uncertainty values
# for SIMS, LA-ICP-MS, and EPMA.


# ============================================================
# 1. PLOTTING PARAMETERS
# ============================================================

# General font sizes.
TITLE_FS = 24
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 16


# ============================================================
# 2. FIGURE 1 CONTROL
# ============================================================

# Keep this figure enabled when running the full script, as the analytical
# uncertainty definitions introduced here are reused by subsequent figures.
run_fig2 = True

if run_fig2:


    # ============================================================
    # 3. ANALYTICAL UNCERTAINTY ASSUMPTIONS
    # ============================================================

    # Representative relative and absolute analytical uncertainties.
    #
    # At every point along the concentration profile, the larger of
    # the relative uncertainty and the absolute uncertainty is used.
    uncertainty_models = {
        "SIMS": {
            "rel_frac": 0.05,
            "abs_ppm": 5.0,
            "color": "red",
            "label": "SIMS"
        },
        "LA_ICP_MS": {
            "rel_frac": 0.08,
            "abs_ppm": 8.0,
            "color": "blue",
            "label": "LA-ICP-MS"
        },
        "EPMA": {
            "rel_frac": 0.15,
            "abs_ppm": 15.0,
            "color": "#00FF00",
            "label": "EPMA"
        },
    }


    # ============================================================
    # 4. UNCERTAINTY ENVELOPE FUNCTION
    # ============================================================

    def build_uncertainty_envelope(
        profile,
        rel_frac,
        abs_ppm
    ):
        """
        Construct the analytical uncertainty envelope around a
        concentration profile.

        The uncertainty at each position is defined as:

            sigma = max(rel_frac * profile, abs_ppm)

        Parameters
        ----------
        profile : array-like
            Concentration profile in ppm.
        rel_frac : float
            Relative analytical uncertainty expressed as a fraction.
        abs_ppm : float
            Absolute analytical uncertainty in ppm.

        Returns
        -------
        sigma : ndarray
            Analytical uncertainty at each position.
        lower : ndarray
            Lower uncertainty bound.
        upper : ndarray
            Upper uncertainty bound.
        """

        sigma = np.maximum(
            rel_frac * profile,
            abs_ppm
        )

        lower = profile - sigma
        upper = profile + sigma

        return sigma, lower, upper


    # ============================================================
    # 5. REFERENCE BA PROFILE
    # ============================================================

    # Manuscript Figure 1 uses the analytical isothermal Ba diffusion profile
    # calculated previously in Manuscript Figure 2.

    profile_for_envelopes = Ba_profile_erf_constT

    profile_label = (
        f"Ba erf-derived "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )


    # ============================================================
    # 6. FIGURE-SPECIFIC FONT SIZES
    # ============================================================

    TITLE_FS = 24
    LABEL_FS = 22
    TICK_FS = 18
    LEGEND_FS = 12


    # ============================================================
    # 7. CREATE FIGURE
    # ============================================================

    fig_env, ax_env = plt.subplots(
        1,
        1,
        figsize=(10, 7.5),
        sharex=True
    )


    # ------------------------------------------------------------
    # Initial concentration profile
    # ------------------------------------------------------------

    ax_env.plot(
        x_um,
        Ba_initial,
        color="black",
        linewidth=3.0,
        linestyle="--",
        label="Initial (t=0)"
    )


    # ------------------------------------------------------------
    # Diffused Ba reference profile
    # ------------------------------------------------------------

    ax_env.plot(
        x_um,
        profile_for_envelopes,
        color="black",
        linewidth=3.0,
        label=profile_label
    )


    # ============================================================
    # 8. SHADED ANALYTICAL UNCERTAINTY ENVELOPES
    # ============================================================

    # The widest envelope is plotted first so that the progressively
    # narrower uncertainty ranges remain visible on top.
    for key in ["EPMA", "LA_ICP_MS", "SIMS"]:

        cfg = uncertainty_models[key]

        # Calculate the analytical uncertainty limits for the
        # current analytical method.
        sigma, lower, upper = build_uncertainty_envelope(
            profile_for_envelopes,
            rel_frac=cfg["rel_frac"],
            abs_ppm=cfg["abs_ppm"]
        )

        # Shaded uncertainty range.
        ax_env.fill_between(
            x_um,
            lower,
            upper,
            color=cfg["color"],
            alpha=0.3,
            label=(
                f"{cfg['label']} "
                f"(±max({cfg['rel_frac']*100:.0f}%, "
                f"{cfg['abs_ppm']:.0f} ppm))"
            )
        )

        # Lower boundary of the uncertainty envelope.
        ax_env.plot(
            x_um,
            lower,
            color=cfg["color"],
            linewidth=1.5
        )

        # Upper boundary of the uncertainty envelope.
        ax_env.plot(
            x_um,
            upper,
            color=cfg["color"],
            linewidth=1.5
        )


    # ============================================================
    # 9. AXES AND FIGURE FORMATTING
    # ============================================================

    ax_env.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    ax_env.set_ylabel(
        "Ba (ppm)",
        fontsize=LABEL_FS
    )

    # Tick-label formatting.
    ax_env.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    # Figure legend.
    ax_env.legend(
        fontsize=LEGEND_FS
    )

    # Background grid.
    ax_env.grid(
        alpha=0.25
    )


    # ============================================================
    # 10. DISPLAY FIGURE
    # ============================================================

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 11. SAVE FIGURE
    # ============================================================

    output_name_env = (
        f"Figure_1_Ba_diffusion_and_analytical_envelopes_"
        f"{t_years:.0f}yr"
    )

    # Save to the Figures directory defined at the beginning
    # of the master script.
    output_file_fig2 = (
        OUTPUT_DIR
        / f"{output_name_env}.png"
    )

    # High-resolution PNG.
    fig_env.savefig(
        output_file_fig2,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

# ============================================================
# END OF MANUSCRIPT FIGURE 1
# ============================================================

#endregion









#region MANUSCRIPT FIGURE 3 — HIDDEN SOLUTIONS UNDER ISOTHERMAL CONDITIONS

import matplotlib.colors as mcolors
from tqdm import tqdm

# ============================================================
# MANUSCRIPT FIGURE 3 — HIDDEN SOLUTIONS UNDER ISOTHERMAL CONDITIONS
# ============================================================

# Manuscript Figure 3 explores the non-uniqueness of diffusion solutions under
# isothermal conditions.
#
# A reference Ba diffusion profile is compared with a large grid of
# analytical diffusion models spanning different temperatures and
# durations. A candidate model is accepted only when its complete
# concentration profile lies within the selected analytical
# uncertainty envelope.
#
# The calculations use the same diffusion parameterisation and
# reference profile defined in the preceding sections of the script.


# ============================================================
# 1. FIGURE SETTINGS
# ============================================================

# Font sizes used for Figure 3 workflow.
TITLE_FS = 22
LABEL_FS = 18
TICK_FS = 18
LEGEND_FS = 14

# Conversion between years and seconds.
seconds_per_year = 365.25 * 24.0 * 3600.0

# Figure control.
run_fig3 = True

# Select the analytical uncertainty envelope used for model acceptance. 
use_EPMA_fig3 = False  # True for EPMA, False for SIMS


if run_fig3:


    # ============================================================
    # 2. ANALYTICAL STEP-DIFFUSION PROFILE
    # ============================================================

    # Analytical error-function solution used to generate each
    # candidate constant-temperature diffusion profile.
    def step_erf_profile(
        x_um,
        t_s,
        D_m2s,
        C_high,
        C_low,
        x0_um=0.0
    ):

        x_m = (x_um - x0_um) * 1e-6

        denom = 2.0 * math.sqrt(
            D_m2s * t_s
        )

        z = x_m / denom

        return (
            C_low
            + 0.5
            * (C_high - C_low)
            * (1.0 - np.vectorize(math.erf)(z))
        )


    # ============================================================
    # 3. ANALYTICAL UNCERTAINTY ENVELOPE
    # ============================================================

    # Rejected profiles are also plotted in Figure 3.
    PLOT_REJECTED_PROFILES_FIG3 = True

    # Select the analytical method used to define the envelope.
    if use_EPMA_fig3:
        selected_method_fig3 = "EPMA"

    else:
        selected_method_fig3 = "SIMS"

    selected_cfg_fig3 = uncertainty_models[
        selected_method_fig3
    ]


    # The reference profile is the constant-temperature analytical
    # Ba profile generated previously.
    reference_profile_fig3 = Ba_profile_erf_constT

    reference_label_fig3 = (
        f"Erf-derived profile "
        f"({t_years:.0f} yr at {T_C:.0f} °C)"
    )


    # Construct the selected analytical uncertainty envelope around
    # the reference Ba profile.
    sigma_ref_fig3, lower_ref_fig3, upper_ref_fig3 = (
        build_uncertainty_envelope(
            reference_profile_fig3,
            rel_frac=selected_cfg_fig3["rel_frac"],
            abs_ppm=selected_cfg_fig3["abs_ppm"]
        )
    )


    # ============================================================
    # 4. TEMPERATURE–TIME SEARCH GRID
    # ============================================================

    # Candidate constant-temperature models span 800–1500 °C.
    T_search_const_C = np.linspace(
        800.0,
        1500.0,
        300
    )

    # Candidate durations span 1–10,000 years on a logarithmic grid.
    time_search_const_yr = np.logspace(
        0,
        4,
        500
    )


    # ============================================================
    # 5. MODEL SEARCH AND ACCEPTANCE
    # ============================================================

    accepted_models_fig3 = []
    rejected_models_fig3 = []

    total_models_fig3 = (
        len(T_search_const_C)
        * len(time_search_const_yr)
    )

    pbar_fig3 = tqdm(
        total=total_models_fig3,
        desc="Figure 3 search",
        unit="model"
    )


    # Explore every temperature–duration combination.
    for T_model_C in T_search_const_C:

        T_model_K = T_model_C + 273.15

        # Diffusion coefficient using the generic function defined earlier in the script. 
        D_model = diffusion_coefficient(
            T_model_K,
            X_An,
            "Ba"
        )

        for t_model_yr in time_search_const_yr:

            t_model_s = (
                t_model_yr
                * seconds_per_year
            )

            # Generate the candidate analytical Ba diffusion profile.
            model_profile = step_erf_profile(
                x_um=x_um,
                t_s=t_model_s,
                D_m2s=D_model,
                C_high=Ba_high,
                C_low=Ba_low,
                x0_um=x0_um
            )

            # A model is accepted only if every point along the
            # profile remains inside the analytical uncertainty
            # envelope of the reference profile.
            is_accepted = np.all(
                (model_profile >= lower_ref_fig3)
                & (model_profile <= upper_ref_fig3)
            )

            record = {
                "T_C": T_model_C,
                "time_yr": t_model_yr,
                "profile": model_profile.copy(),
                "D_m2s": D_model
            }

            if is_accepted:

                accepted_models_fig3.append(
                    record
                )

            else:

                rejected_models_fig3.append(
                    record
                )

            pbar_fig3.update(1)


    pbar_fig3.close()


    # ============================================================
    # 6. PREPARE DATA FOR PLOTTING
    # ============================================================

    # Extract temperatures and durations of accepted models.
    if len(accepted_models_fig3) > 0:

        T_plot_fig3 = np.array(
            [
                m["T_C"]
                for m in accepted_models_fig3
            ]
        )

        t_plot_fig3 = np.array(
            [
                m["time_yr"]
                for m in accepted_models_fig3
            ]
        )

    else:

        T_plot_fig3 = np.array([])
        t_plot_fig3 = np.array([])


    # ------------------------------------------------------------
    # Thin accepted profiles for plotting
    # ------------------------------------------------------------

    # The complete accepted-model population is retained for the
    # calculations. 
    max_profiles_to_plot_fig3 = 500

    # Fixed random seed retained for reproducible profile selection.
    rng_fig3 = np.random.default_rng(42)

    profiles_to_plot_fig3 = (
        accepted_models_fig3.copy()
    )

    if (
        len(profiles_to_plot_fig3)
        > max_profiles_to_plot_fig3
    ):

        idx = rng_fig3.choice(
            len(profiles_to_plot_fig3),
            size=max_profiles_to_plot_fig3,
            replace=False
        )

        profiles_to_plot_fig3 = [
            profiles_to_plot_fig3[i]
            for i in idx
        ]


    # ------------------------------------------------------------
    # Thin rejected profiles for plotting
    # ------------------------------------------------------------

    max_profiles_rejected_fig3 = 2000

    profiles_rejected_plot_fig3 = (
        rejected_models_fig3.copy()
    )

    if (
        len(profiles_rejected_plot_fig3)
        > max_profiles_rejected_fig3
    ):

        idx = rng_fig3.choice(
            len(profiles_rejected_plot_fig3),
            size=max_profiles_rejected_fig3,
            replace=False
        )

        profiles_rejected_plot_fig3 = [
            profiles_rejected_plot_fig3[i]
            for i in idx
        ]


    # ============================================================
    # 7. SUMMARY STATISTICS
    # ============================================================

    n_tested_fig3 = (
        len(T_search_const_C)
        * len(time_search_const_yr)
    )

    n_accepted_fig3 = len(
        accepted_models_fig3
    )

    accepted_pct_fig3 = (
        100
        * n_accepted_fig3
        / n_tested_fig3
        if n_tested_fig3 > 0
        else 0.0
    )


    if len(accepted_models_fig3) > 0:

        Tmin_acc_fig3 = np.min(
            T_plot_fig3
        )

        Tmax_acc_fig3 = np.max(
            T_plot_fig3
        )

        # Accepted time range.
        tmin_acc_fig3 = np.min(
            [
                m["time_yr"]
                for m in accepted_models_fig3
            ]
        )

        tmax_acc_fig3 = np.max(
            [
                m["time_yr"]
                for m in accepted_models_fig3
            ]
        )

    else:

        Tmin_acc_fig3 = np.nan
        Tmax_acc_fig3 = np.nan


    # ============================================================
    # 8. CREATE FIGURE 3
    # ============================================================

    fig3, (axMain, axRight) = plt.subplots(
        2,
        1,
        figsize=(10.625, 15.0),
        gridspec_kw={
            "height_ratios": [1.25, 1.0]
        }
    )


    # ============================================================
    # 9. PANEL A — DIFFUSION PROFILES
    # ============================================================

    # Upper analytical uncertainty limit.
    axMain.plot(
        x_um,
        upper_ref_fig3,
        color="red",
        linewidth=1.6,
        linestyle="--",
        label=(
            f"{selected_cfg_fig3['label']} limits"
        )
    )

    # Lower analytical uncertainty limit.
    axMain.plot(
        x_um,
        lower_ref_fig3,
        color="red",
        linewidth=1.6,
        linestyle="--",
        label="_nolegend_"
    )


    # ------------------------------------------------------------
    # Accepted models
    # ------------------------------------------------------------

    if len(profiles_to_plot_fig3) > 0:

        # Accepted profiles are coloured according to their duration.
        norm_profiles_fig3 = mcolors.LogNorm(
            vmin=np.min(
                time_search_const_yr
            ),
            vmax=np.max(
                time_search_const_yr
            )
        )

        cmap_profiles_fig3 = plt.cm.viridis

        first_line = True

        for model in profiles_to_plot_fig3:

            if first_line:

                label_models = (
                    f"Accepted models: "
                    f"{accepted_pct_fig3:.1f}% "
                    f"({n_accepted_fig3}/{n_tested_fig3})"
                )

            else:

                label_models = "_nolegend_"


            axMain.plot(
                x_um,
                model["profile"],
                color=cmap_profiles_fig3(
                    norm_profiles_fig3(
                        model["time_yr"]
                    )
                ),
                linewidth=1.3,
                alpha=1.0,
                label=label_models,
                zorder=2
            )

            first_line = False


    # ------------------------------------------------------------
    # Rejected models
    # ------------------------------------------------------------

    if (
        PLOT_REJECTED_PROFILES_FIG3
        and len(profiles_rejected_plot_fig3) > 0
    ):

        first_rejected = True

        for model in profiles_rejected_plot_fig3:

            label_rejected = (
                "Rejected models"
                if first_rejected
                else "_nolegend_"
            )

            axMain.plot(
                x_um,
                model["profile"],
                color="grey",
                linewidth=0.1,
                alpha=0.2,
                label=label_rejected,
                zorder=1
            )

            first_rejected = False


    # ------------------------------------------------------------
    # Reference and initial profiles
    # ------------------------------------------------------------

    axMain.plot(
        x_um,
        reference_profile_fig3,
        color="black",
        linewidth=4.0,
        label=reference_label_fig3
    )

    axMain.plot(
        x_um,
        Ba_initial,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Initial (t=0)"
    )


    # Panel label.
    axMain.text(
        0.98,
        0.98,
        "a",
        transform=axMain.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    # Axes and formatting.
    axMain.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    axMain.set_ylabel(
        "Ba (ppm)",
        fontsize=LABEL_FS
    )

    axMain.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axMain.grid(
        alpha=0.25
    )

    axMain.set_xlim(
        -200,
        200
    )

    legend3 = axMain.legend(
        fontsize=LEGEND_FS,
        loc="lower left"
    )

    legend3.get_frame().set_linewidth(
        1.0
    )

    legend3.get_frame().set_edgecolor(
        "black"
    )


    # ============================================================
    # 10. ACCEPTED TEMPERATURE–DURATION SOLUTIONS
    # ============================================================

    if len(accepted_models_fig3) > 0:

        T_acc_panel = np.array(
            [
                m["T_C"]
                for m in accepted_models_fig3
            ]
        )

        t_acc_panel = np.array(
            [
                m["time_yr"]
                for m in accepted_models_fig3
            ]
        )


        # Accepted solutions are coloured according to duration.
        sc3 = axRight.scatter(
            T_acc_panel,
            t_acc_panel,
            c=t_acc_panel,
            cmap="viridis",
            norm=mcolors.LogNorm(
                vmin=np.min(
                    time_search_const_yr
                ),
                vmax=np.max(
                    time_search_const_yr
                )
            ),
            s=36,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85,
            label=(
                f"Accepted solutions "
                f"({n_accepted_fig3})"
            )
        )


        cbar3 = plt.colorbar(
            sc3,
            ax=axRight,
            fraction=0.046,
            pad=0.03
        )

        cbar3.set_label(
            "Accepted duration (yr)",
            fontsize=LABEL_FS
        )

        cbar3.ax.tick_params(
            labelsize=TICK_FS
        )


    else:

        axRight.text(
            0.5,
            0.5,
            "No accepted models",
            transform=axRight.transAxes,
            ha="center",
            va="center",
            fontsize=LEGEND_FS
        )


    # ------------------------------------------------------------
    # Reference model
    # ------------------------------------------------------------

    axRight.scatter(
        [T_C],
        [t_years],
        marker="*",
        s=300,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label=(
            f"Reference "
            f"({T_C:.0f} °C, {t_years:.0f} yr)"
        ),
        zorder=5
    )


    # ------------------------------------------------------------
    # Explored model domain
    # ------------------------------------------------------------

    axRight.set_xlim(
        np.min(T_search_const_C),
        np.max(T_search_const_C)
    )

    axRight.set_ylim(
        np.min(time_search_const_yr),
        np.max(time_search_const_yr)
    )

    # Time was searched logarithmically.
    axRight.set_yscale(
        "log"
    )


    # ------------------------------------------------------------
    # Search-domain information
    # ------------------------------------------------------------

    search_text_fig3 = (
        "Search domain:\n"
        rf"$T$ = "
        rf"{np.min(T_search_const_C):.0f}–"
        rf"{np.max(T_search_const_C):.0f} °C"
        "\n"
        rf"$t$ = "
        rf"{np.min(time_search_const_yr):.0f}–"
        rf"{np.max(time_search_const_yr):.0f} yr"
        "\n\n"
        "Accepted range:\n"
        rf"$T$ = "
        rf"{Tmin_acc_fig3:.1f}–"
        rf"{Tmax_acc_fig3:.1f} °C"
        "\n"
        rf"$t$ = "
        rf"{tmin_acc_fig3:.1f}–"
        rf"{tmax_acc_fig3:.0f} yr"
    )


    axRight.text(
        0.97,
        0.03,
        search_text_fig3,
        transform=axRight.transAxes,
        ha="right",
        va="bottom",
        fontsize=LEGEND_FS,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.92
        )
    )


    # Panel label.
    axRight.text(
        0.98,
        0.98,
        "b",
        transform=axRight.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    # Axes and formatting.
    axRight.set_xlabel(
        "Temperature (°C)",
        fontsize=LABEL_FS
    )

    axRight.set_ylabel(
        "Duration (yr)",
        fontsize=LABEL_FS
    )


    axRight.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axRight.grid(
        alpha=0.25
    )

    axRight.legend(
        fontsize=LEGEND_FS,
        loc="best"
    )


    # ============================================================
    # 11. DISPLAY FIGURE
    # ============================================================

    plt.tight_layout(
        h_pad=0.5
    )

    plt.show()


    # ============================================================
    # 12. SAVE FIGURE
    # ============================================================

    output_name_fig3 = (
        f"Figure_3_constantT_"
        f"{selected_method_fig3}_within envelope_"
        f"{t_years:.0f}yr"
    )

    # Save to the common Figures directory established at the beginning of the master script.
    output_file_fig3 = (
        OUTPUT_DIR
        / f"{output_name_fig3}.png"
    )

    fig3.savefig(
        output_file_fig3,
        dpi=300,
        bbox_inches="tight"
    )


# ============================================================
# END OF MANUSCRIPT FIGURE 3
# ============================================================
#endregion










#region MANUSCRIPT FIGURE 4 — NON-ISOTHERMAL TEMPERATURE–TIME SOLUTIONS
# ============================================================
# MANUSCRIPT FIGURE 4 — NON-ISOTHERMAL TEMPERATURE–TIME SOLUTIONS
# ============================================================

# Manuscript Figure 4 explores the range of non-isothermal thermal histories
# that can reproduce the reference Ba diffusion profile within the
# selected analytical uncertainty envelope.
#
# Candidate models follow linear temperature paths between an initial
# and final temperature. For each thermal path and duration, diffusivity
# is integrated through time and used to calculate the corresponding
# analytical diffusion profile.
#
# A model is accepted only when the complete calculated profile falls
# within the analytical uncertainty envelope of the reference profile.


# ============================================================
# 1. FIGURE SETTINGS
# ============================================================

# Figure 4 font sizes.
TITLE_FS = 20
LABEL_FS = 16
TICK_FS = 13
LEGEND_FS = 9

# Figure control.
run_fig4 = True

# Select the analytical uncertainty envelope used for model acceptance.
use_EPMA_fig4 = False # True for EPMA, False for SIMS


if run_fig4:


    # ============================================================
    # 2. ANALYTICAL UNCERTAINTY ENVELOPE
    # ============================================================

    # Plot rejected profiles together with the accepted solutions.
    PLOT_REJECTED_PROFILES = True


    if use_EPMA_fig4:

        selected_method = "EPMA"

    else:

        selected_method = "SIMS"


    selected_cfg = uncertainty_models[
        selected_method
    ]


    # The non-isothermal Ba profile calculated previously is used as the reference profile.
    reference_profile = Ba_profile_erf_variableT


    reference_label = (
        f"Erf-like integrated D(T) "
        f"({t_years:.0f} yr; "
        f"T {Tinitial_Model:.0f}→{Tfinal_Model:.0f} °C)"
    )


    # Construct the uncertainty envelope around the reference profile.
    sigma_ref, lower_ref, upper_ref = build_uncertainty_envelope(
        reference_profile,
        rel_frac=selected_cfg["rel_frac"],
        abs_ppm=selected_cfg["abs_ppm"]
    )


    # ============================================================
    # 3. TEMPERATURE–TIME SEARCH GRID
    # ============================================================

    # Initial and final temperatures independently span 800–1300 °C.
    Tinitial_search_C = np.linspace(
        800.0,
        1300.0,
        50
    )

    Tfinal_search_C = np.linspace(
        800.0,
        1300.0,
        50
    )

    # Model durations span 1–10,000 years.
    time_search_yr = np.logspace(
        0,
        4,
        140
    )


    # ============================================================
    # 4. MODEL SEARCH AND ACCEPTANCE
    # ============================================================

    accepted_models = []
    rejected_models = []


    total_models_fig4 = (
        len(Tinitial_search_C)
        * len(Tfinal_search_C)
        * len(time_search_yr)
    )


    pbar_fig4 = tqdm(
        total=total_models_fig4,
        desc="Figure 4 search",
        unit="model"
    )


    # Explore every initial-temperature, final-temperature, and duration combination in the search grid.
    for Ti in Tinitial_search_C:

        for Tf in Tfinal_search_C:

            for t_model_yr in time_search_yr:


                t_model_s = (
                    t_model_yr
                    * seconds_per_year
                )


                # Linear thermal history for the candidate model.
                t_series_model = np.linspace(
                    0.0,
                    t_model_s,
                    n_steps_integral
                )


                T_series_C_model = (
                    Ti
                    + (Tf - Ti)
                    * (t_series_model / t_model_s)
                )


                T_series_K_model = (
                    T_series_C_model
                    + 273.15
                )


                # Ba diffusivity along the complete thermal history.
                D_Ba_series_model = diffusion_coefficient(
                    T_series_K_model,
                    X_An,
                    "Ba"
                )


                # Integrated diffusivity for the candidate thermal path.
                I_model = np.trapz(
                    D_Ba_series_model,
                    t_series_model
                )


                # Calculate the corresponding Ba diffusion profile.
                model_profile = erf_profile_from_integral(
                    x_um,
                    I_model,
                    Ba_high,
                    Ba_low,
                    x0_um
                )


                # Strict acceptance criterion: the entire model profile
                # must remain inside the analytical uncertainty envelope.
                is_accepted = np.all(
                    (model_profile >= lower_ref)
                    & (model_profile <= upper_ref)
                )


                if is_accepted:

                    accepted_models.append(
                        {
                            "Ti_C": Ti,
                            "Tf_C": Tf,
                            "time_yr": t_model_yr,
                            "profile": model_profile.copy(),
                            "I_m2": I_model
                        }
                    )

                else:

                    rejected_models.append(
                        {
                            "Ti_C": Ti,
                            "Tf_C": Tf,
                            "time_yr": t_model_yr,
                            "profile": model_profile.copy(),
                            "I_m2": I_model
                        }
                    )


                pbar_fig4.update(1)


    pbar_fig4.close()


    print(
        f"\nAccepted models within "
        f"{selected_cfg['label']} envelope: "
        f"{len(accepted_models)}"
    )


    # ============================================================
    # 5. PREPARE ACCEPTED-MODEL DATA
    # ============================================================

    if len(accepted_models) > 0:

        Ti_plot = np.array(
            [
                m["Ti_C"]
                for m in accepted_models
            ]
        )

        Tf_plot = np.array(
            [
                m["Tf_C"]
                for m in accepted_models
            ]
        )

        t_plot = np.array(
            [
                m["time_yr"]
                for m in accepted_models
            ]
        )


        Ti_min_acc = np.min(
            Ti_plot
        )

        Ti_max_acc = np.max(
            Ti_plot
        )


        Tf_min_acc = np.min(
            Tf_plot
        )

        Tf_max_acc = np.max(
            Tf_plot
        )


        t_min_acc = np.min(
            t_plot
        )

        t_max_acc = np.max(
            t_plot
        )


    else:

        Ti_plot = np.array([])
        Tf_plot = np.array([])
        t_plot = np.array([])


        Ti_min_acc = np.nan
        Ti_max_acc = np.nan


        Tf_min_acc = np.nan
        Tf_max_acc = np.nan


        t_min_acc = np.nan
        t_max_acc = np.nan


    # ============================================================
    # 6. PROFILES USED FOR PLOTTING
    # ============================================================

    # Only the displayed profiles are subsampled. All accepted and
    # rejected models remain available for the calculations.
    max_profiles_to_plot_fig4 = 500

    rng = np.random.default_rng(
        42
    )


    profiles_to_plot_fig4 = (
        accepted_models.copy()
    )


    if (
        len(profiles_to_plot_fig4)
        > max_profiles_to_plot_fig4
    ):

        idx = rng.choice(
            len(profiles_to_plot_fig4),
            size=max_profiles_to_plot_fig4,
            replace=False
        )

        profiles_to_plot_fig4 = [
            profiles_to_plot_fig4[i]
            for i in idx
        ]


    # Rejected profiles displayed in panel a.
    max_profiles_rejected = 2000


    profiles_rejected_plot = (
        rejected_models.copy()
    )


    if (
        len(profiles_rejected_plot)
        > max_profiles_rejected
    ):

        idx = rng.choice(
            len(profiles_rejected_plot),
            size=max_profiles_rejected,
            replace=False
        )

        profiles_rejected_plot = [
            profiles_rejected_plot[i]
            for i in idx
        ]


    # ============================================================
    # 7. CREATE FIGURE 4
    # ============================================================

    fig4, (axL, axR) = plt.subplots(
        2,
        1,
        figsize=(8.5, 12.0),
        gridspec_kw={
            "height_ratios": [1.25, 1.0]
        }
    )


    # ============================================================
    # 8. PANEL A — DIFFUSION PROFILES
    # ============================================================

    # Upper analytical uncertainty limit.
    axL.plot(
        x_um,
        upper_ref,
        color="red",
        linewidth=1.6,
        linestyle="--",
        label=f"{selected_cfg['label']} limits"
    )


    # Lower analytical uncertainty limit.
    axL.plot(
        x_um,
        lower_ref,
        color="red",
        linewidth=1.6,
        linestyle="--",
        label="_nolegend_"
    )


    # Number of explored and accepted models.
    n_tested = (
        len(Tinitial_search_C)
        * len(Tfinal_search_C)
        * len(time_search_yr)
    )

    n_accepted = len(
        accepted_models
    )


    # ------------------------------------------------------------
    # Accepted profiles
    # ------------------------------------------------------------

    if len(profiles_to_plot_fig4) > 0:

        norm_profiles = mcolors.LogNorm(
            vmin=np.min(
                time_search_yr
            ),
            vmax=np.max(
                time_search_yr
            )
        )

        cmap_profiles = plt.cm.viridis


        first_line = True


        for model in profiles_to_plot_fig4:


            if first_line:

                label_models = (
                    f"Accepted models: "
                    f"{100*n_accepted/n_tested:.1f}% "
                    f"({n_accepted}/{n_tested})"
                )

            else:

                label_models = "_nolegend_"


            axL.plot(
                x_um,
                model["profile"],
                color=cmap_profiles(
                    norm_profiles(
                        model["time_yr"]
                    )
                ),
                linewidth=1.3,
                alpha=1.0,
                label=label_models,
                zorder=2
            )


            first_line = False


    # ------------------------------------------------------------
    # Rejected profiles
    # ------------------------------------------------------------

    if (
        PLOT_REJECTED_PROFILES
        and len(profiles_rejected_plot) > 0
    ):


        first_rejected = True


        for model in profiles_rejected_plot:


            label_rejected = (
                "Rejected models"
                if first_rejected
                else "_nolegend_"
            )


            axL.plot(
                x_um,
                model["profile"],
                color="grey",
                linewidth=0.1,
                alpha=0.2,
                label=label_rejected,
                zorder=1
            )


            first_rejected = False


    # ------------------------------------------------------------
    # Reference profile
    # ------------------------------------------------------------

    axL.plot(
        x_um,
        reference_profile,
        color="black",
        linewidth=4.0,
        label=reference_label
    )


    # ------------------------------------------------------------
    # Initial profile
    # ------------------------------------------------------------

    axL.plot(
        x_um,
        Ba_initial,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Initial (t=0)"
    )


    # Panel label.
    axL.text(
        0.98,
        0.98,
        "a",
        transform=axL.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    # Axes and formatting.
    axL.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    axL.set_ylabel(
        "Ba (ppm)",
        fontsize=LABEL_FS
    )


    axL.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )


    axL.set_xlim(
        -200,
        200
    )


    axL.grid(
        alpha=0.25
    )


    legend = axL.legend(
        fontsize=LEGEND_FS,
        loc="lower left"
    )


    legend.get_frame().set_linewidth(
        1.0
    )

    legend.get_frame().set_edgecolor(
        "black"
    )


    # ============================================================
    # 9. PANEL B — ACCEPTED THERMAL HISTORIES
    # ============================================================

    if len(accepted_models) > 0:


        # Accepted models are plotted in initial- versus final-
        # temperature space and coloured according to duration.
        sc = axR.scatter(
            Ti_plot,
            Tf_plot,
            c=t_plot,
            cmap="viridis",
            norm=mcolors.LogNorm(
                vmin=np.min(
                    time_search_yr
                ),
                vmax=np.max(
                    time_search_yr
                )
            ),
            s=36,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85
        )


        # Reference thermal history.
        axR.scatter(
            [Tinitial_Model],
            [Tfinal_Model],
            marker="*",
            s=240,
            color="red",
            edgecolor="black",
            linewidth=0.8,
            label=(
                fr"Reference "
                fr"({Tinitial_Model:.0f}→"
                fr"{Tfinal_Model:.0f} °C)"
            )
        )


        # Colour bar showing accepted model duration.
        cbarR = plt.colorbar(
            sc,
            ax=axR,
            fraction=0.046,
            pad=0.03
        )


        cbarR.set_label(
            "Accepted duration (yr)",
            fontsize=LABEL_FS
        )


        cbarR.ax.tick_params(
            labelsize=TICK_FS
        )


    else:


        axR.text(
            0.5,
            0.5,
            "No accepted models",
            transform=axR.transAxes,
            ha="center",
            va="center",
            fontsize=LEGEND_FS
        )


    # ============================================================
    # 10. SEARCH-DOMAIN INFORMATION
    # ============================================================

    search_text_fig4 = (
        "Search domain:\n"
        rf"$T_i$ = "
        rf"{np.min(Tinitial_search_C):.0f}–"
        rf"{np.max(Tinitial_search_C):.0f} °C"
        "\n"
        rf"$T_f$ = "
        rf"{np.min(Tfinal_search_C):.0f}–"
        rf"{np.max(Tfinal_search_C):.0f} °C"
        "\n"
        rf"$t$ = "
        rf"{np.min(time_search_yr):.0f}–"
        rf"{np.max(time_search_yr):.0f} yr"
        "\n\n"
        "Accepted range:\n"
        rf"$T_i$ = "
        rf"{Ti_min_acc:.1f}–"
        rf"{Ti_max_acc:.1f} °C"
        "\n"
        rf"$T_f$ = "
        rf"{Tf_min_acc:.1f}–"
        rf"{Tf_max_acc:.1f} °C"
        "\n"
        rf"$t$ = "
        rf"{t_min_acc:.1f}–"
        rf"{t_max_acc:.0f} yr"
    )


    axR.text(
        0.03,
        0.97,
        search_text_fig4,
        transform=axR.transAxes,
        ha="left",
        va="top",
        fontsize=LEGEND_FS,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.92
        )
    )


    # Panel label.
    axR.text(
        0.98,
        0.98,
        "b",
        transform=axR.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    # Axes and formatting.
    axR.set_xlabel(
        "$T_{{initial}}$ (°C)",
        fontsize=LABEL_FS
    )

    axR.set_ylabel(
        "$T_{{final}}$ (°C)",
        fontsize=LABEL_FS
    )


    axR.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )


    axR.grid(
        alpha=0.25
    )


    axR.legend(
        fontsize=LEGEND_FS,
        loc="lower left"
    )


    # ============================================================
    # 11. DISPLAY FIGURE
    # ============================================================

    plt.tight_layout(
        h_pad=1.5
    )

    plt.show()


    # ============================================================
    # 12. SAVE FIGURE
    # ============================================================

    output_name_fig4 = (
        f"Figure_4_Non-isothermal_"
        f"{selected_method}_envelope"
    )


    output_file_fig4 = (
        OUTPUT_DIR
        / f"{output_name_fig4}.png"
    )


    fig4.savefig(
        output_file_fig4,
        dpi=300,
        bbox_inches="tight"
    )


# ============================================================
# END OF FIGURE 4
# ============================================================
#endregion










#region MANUSCRIPT FIGURE 5 — HIDDEN RESIDENCE TIME
# ============================================================
# MANUSCRIPT FIGURE 5 — HIDDEN RESIDENCE TIME
# ============================================================

# Manuscript Figure 5 evaluates the amount of additional 
# residence time that can remain analytically undetectable 
# following or preceding a reference thermal history.
#
# Calculations are performed for Sr and Ba across different
# plagioclase anorthite fractions. A common analytical uncertainty
# envelope anchored at XAn = 0.50 is used to compare the effect of
# composition on the maximum hidden residence time.


# ============================================================
# 1. FIGURE SETTINGS
# ============================================================

TITLE_FS = 14
LABEL_FS = 13
TICK_FS = 12
LEGEND_FS = 11

# Figure control.
run_fig5 = True
use_EPMA_fig5 = False # True for EPMA, False for SIMS


if run_fig5:


    # ============================================================
    # 2. ANALYTICAL UNCERTAINTY
    # ============================================================

    selected_method_fig5 = (
        "EPMA"
        if use_EPMA_fig5
        else "SIMS"
    )

    selected_cfg_fig5 = uncertainty_models[
        selected_method_fig5
    ]

    uncertainty_rel_frac_fig5 = (
        selected_cfg_fig5["rel_frac"]
    )

    uncertainty_abs_ppm_fig5 = (
        selected_cfg_fig5["abs_ppm"]
    )



    # ============================================================
    # 3. MODEL PARAMETERS
    # ============================================================

    # Plagioclase compositions explored.
    XAn_values_fig5 = [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00
    ]

    # Common analytical-envelope anchor.
    XAn_anchor = 0.50

    # Residence-temperature range and number of evaluated temperatures.
    deltaT_res_fig5 = 150.0
    n_Tres_fig5 = 5

    # Finite-difference and bisection parameters.
    t_high_init_yr = 10.0
    t_high_max_yr = 1e8
    tol_rel_time = 0.05
    alpha_safety = 0.4


    # Detectability window around the diffusion interface.
    x_window_um = 80.0

    # Horizontal range displayed for the diffusion profiles.
    xlim_profiles_fig5 = (
        -200,
        200
    )


    # ============================================================
    # 4. REFERENCE THERMAL HISTORIES
    # ============================================================

    # Cooling reference history.
    Tinitial_ref_fig5 = Tinitial_Model
    Tfinal_ref_fig5 = Tfinal_Model
    t_ref_s_fig5 = t_s


    # Heating reference history.
    Tinitial_ref2_fig5 = 900.0
    Tfinal_ref2_fig5 = 1000.0
    t_ref2_yr_fig5 = 3000.0

    t_ref2_s_fig5 = (
        t_ref2_yr_fig5
        * seconds_per_year
    )


    # Residence-temperature arrays.
    T_cooling_fig5 = np.linspace(
        Tfinal_ref_fig5,
        Tfinal_ref_fig5 - deltaT_res_fig5,
        n_Tres_fig5
    )

    T_warming_fig5 = np.linspace(
        Tinitial_ref_fig5,
        Tinitial_ref_fig5 + deltaT_res_fig5,
        n_Tres_fig5
    )


    # ============================================================
    # 5. HELPER FUNCTIONS
    # ============================================================

    def build_max_envelope(
        profile,
        rel_frac,
        abs_ppm
    ):

        sigma = np.maximum(
            rel_frac * profile,
            abs_ppm
        )

        return (
            sigma,
            profile - sigma,
            profile + sigma
        )


    def erf_like_reference_profile_fig5(
        species,
        XAn_value,
        C_high,
        C_low
    ):

        t_series_local = np.linspace(
            0.0,
            t_ref_s_fig5,
            n_steps_integral
        )

        T_series_C_local = (
            Tinitial_ref_fig5
            + (
                Tfinal_ref_fig5
                - Tinitial_ref_fig5
            )
            * (
                t_series_local
                / t_ref_s_fig5
            )
        )

        T_series_K_local = (
            T_series_C_local
            + 273.15
        )

        D_series_local = diffusion_coefficient(
            T_series_K_local,
            XAn_value,
            species
        )

        I_ref_local = np.trapz(
            D_series_local,
            t_series_local
        )

        C_ref_local = erf_profile_from_integral(
            x_um,
            I_ref_local,
            C_high,
            C_low,
            x0_um
        )

        return (
            I_ref_local,
            C_ref_local
        )


    def erf_like_profile_custom_path_fig5(
        species,
        XAn_value,
        C_high,
        C_low,
        Ti_C,
        Tf_C,
        t_path_s
    ):

        t_series_local = np.linspace(
            0.0,
            t_path_s,
            n_steps_integral
        )

        T_series_C_local = (
            Ti_C
            + (Tf_C - Ti_C)
            * (
                t_series_local
                / t_path_s
            )
        )

        T_series_K_local = (
            T_series_C_local
            + 273.15
        )

        D_series_local = diffusion_coefficient(
            T_series_K_local,
            XAn_value,
            species
        )

        I_ref_local = np.trapz(
            D_series_local,
            t_series_local
        )

        C_ref_local = erf_profile_from_integral(
            x_um,
            I_ref_local,
            C_high,
            C_low,
            x0_um
        )

        return (
            I_ref_local,
            C_ref_local
        )


    def inside_envelope_window(
        C,
        lower,
        upper,
        x_um,
        x_window=None
    ):

        if x_window is None:

            m = slice(None)

        else:

            m = (
                (x_um >= -x_window)
                & (x_um <= x_window)
            )

        return np.all(
            (C[m] >= lower[m])
            & (C[m] <= upper[m])
        )


    def evolve_fd_constT_to_time(
        C0,
        t_target_s,
        D_const,
        x_um,
        alpha_safety=0.4
    ):

        if (
            D_const <= 0.0
            or t_target_s <= 0.0
        ):

            return (
                C0.astype(float).copy()
            )


        x_m = x_um * 1e-6
        dx = x_m[1] - x_m[0]

        dt_stable = (
            alpha_safety
            * dx
            * dx
            / D_const
        )

        n_steps = max(
            1,
            int(
                np.ceil(
                    t_target_s
                    / dt_stable
                )
            )
        )

        dt_eff = (
            t_target_s
            / n_steps
        )

        alpha = (
            D_const
            * dt_eff
            / (dx * dx)
        )


        C = C0.astype(float).copy()


        for _ in range(n_steps):

            Cn = C
            C = Cn.copy()

            C[1:-1] = (
                Cn[1:-1]
                + alpha
                * (
                    Cn[2:]
                    - 2.0 * Cn[1:-1]
                    + Cn[:-2]
                )
            )

            C[0] = C[1]
            C[-1] = C[-2]


        return C


    # ============================================================
    # 6. CRITICAL RESIDENCE TIME BY BISECTION
    # ============================================================

    def tcrit_bisection_fd(
        Cstart,
        Tres_C,
        XAn_value,
        species,
        lower,
        upper,
        x_um,
        t_high_init_yr=10.0,
        t_high_max_yr=1e6,
        tol_rel=0.01,
        x_window=None,
        alpha_safety=0.4
    ):

        D_const = diffusion_coefficient(
            Tres_C + 273.15,
            XAn_value,
            species
        )


        if D_const <= 0.0:

            return 0.0


        sec_per_yr = (
            365.25
            * 24.0
            * 3600.0
        )

        t_low_s = 0.0

        t_high_s = (
            t_high_init_yr
            * sec_per_yr
        )


        # ------------------------------------------------------------
        # Bracket the critical residence time
        # ------------------------------------------------------------

        while True:

            C_test = evolve_fd_constT_to_time(
                Cstart,
                t_high_s,
                D_const,
                x_um,
                alpha_safety=alpha_safety
            )

            if not inside_envelope_window(
                C_test,
                lower,
                upper,
                x_um,
                x_window
            ):

                break


            t_low_s = t_high_s
            t_high_s *= 2.0


            if (
                t_high_s
                > t_high_max_yr
                * sec_per_yr
            ):

                return (
                    t_high_s
                    / sec_per_yr
                )


        # ------------------------------------------------------------
        # Bisection search
        # ------------------------------------------------------------

        while (
            t_high_s - t_low_s
        ) > (
            tol_rel
            * max(t_high_s, 1.0)
        ):

            t_mid = (
                0.5
                * (
                    t_low_s
                    + t_high_s
                )
            )

            C_mid = evolve_fd_constT_to_time(
                Cstart,
                t_mid,
                D_const,
                x_um,
                alpha_safety=alpha_safety
            )

            if inside_envelope_window(
                C_mid,
                lower,
                upper,
                x_um,
                x_window
            ):

                t_low_s = t_mid

            else:

                t_high_s = t_mid


        return (
            t_low_s
            / sec_per_yr
        )


    # ============================================================
    # 7. HIDDEN RESIDENCE BEFORE A HEATING PATH
    # ============================================================

    def tcrit_hidden_erf_then_path_fd(
        Tres_C,
        XAn_value,
        species,
        C_high,
        C_low,
        lower,
        upper,
        Ti_path_C,
        Tf_path_C,
        t_path_s,
        x_um,
        t_high_init_yr=10.0,
        t_high_max_yr=1e6,
        tol_rel=0.01,
        x_window=None
    ):
        """
        Hidden residence is treated analytically because it is
        isothermal and XAn is constant. The subsequent heating
        path is treated with finite differences.
        """

        sec_per_yr = (
            365.25
            * 24.0
            * 3600.0
        )


        D_hidden = diffusion_coefficient(
            Tres_C + 273.15,
            XAn_value,
            species
        )


        if D_hidden <= 0.0:

            return 0.0


        # The subsequent heating path is fixed at the anchor XAn,
        # isolating the compositional sensitivity of hidden residence.
        XAn_path_value = XAn_anchor


        XAn_profile_local = np.full_like(
            x_um,
            XAn_path_value,
            dtype=float
        )


        # ------------------------------------------------------------
        # Stable timestep for the heating path
        # ------------------------------------------------------------

        x_m = x_um * 1e-6
        dx = x_m[1] - x_m[0]


        T_test_K = np.linspace(
            Ti_path_C + 273.15,
            Tf_path_C + 273.15,
            500
        )


        D_test = diffusion_coefficient(
            T_test_K,
            XAn_path_value,
            species
        )


        D_max_path = np.max(
            D_test
        )


        alpha_limit_path = 0.40


        n_steps_path = int(
            np.ceil(
                D_max_path
                * t_path_s
                / (
                    alpha_limit_path
                    * dx**2
                )
            )
        ) + 2


        t_series_path = np.linspace(
            0.0,
            t_path_s,
            n_steps_path
        )


        T_series_path_C = (
            Ti_path_C
            + (
                Tf_path_C
                - Ti_path_C
            )
            * (
                t_series_path
                / t_path_s
            )
        )


        T_series_path_K = (
            T_series_path_C
            + 273.15
        )


        def final_profile_after_hidden(
            t_hidden_s
        ):

            I_hidden = (
                D_hidden
                * t_hidden_s
            )


            C_hidden = erf_profile_from_integral(
                x_um,
                I_hidden,
                C_high,
                C_low,
                x0_um
            )


            C_final = diffuse_1D_variable_XAn(
                T_series_path_K,
                t_path_s,
                x_um,
                C_hidden,
                XAn_profile_local,
                species
            )


            return C_final


        # ------------------------------------------------------------
        # Bracket the hidden residence time
        # ------------------------------------------------------------

        t_low_s = 0.0

        t_high_s = (
            t_high_init_yr
            * sec_per_yr
        )


        while True:

            C_test = (
                final_profile_after_hidden(
                    t_high_s
                )
            )


            if not inside_envelope_window(
                C_test,
                lower,
                upper,
                x_um,
                x_window
            ):

                break


            t_low_s = t_high_s
            t_high_s *= 2.0


            if (
                t_high_s
                > t_high_max_yr
                * sec_per_yr
            ):

                return (
                    t_high_s
                    / sec_per_yr
                )


        # ------------------------------------------------------------
        # Bisection search
        # ------------------------------------------------------------

        while (
            t_high_s
            - t_low_s
        ) > (
            tol_rel
            * max(t_high_s, 1.0)
        ):

            t_mid_s = (
                0.5
                * (
                    t_low_s
                    + t_high_s
                )
            )


            C_mid = (
                final_profile_after_hidden(
                    t_mid_s
                )
            )


            if inside_envelope_window(
                C_mid,
                lower,
                upper,
                x_um,
                x_window
            ):

                t_low_s = t_mid_s

            else:

                t_high_s = t_mid_s


        return (
            t_low_s
            / sec_per_yr
        )


    # ============================================================
    # 8. REFERENCE PROFILES AND COMMON ENVELOPES
    # ============================================================

    fig5_data = {
        "Sr": {},
        "Ba": {}
    }


    for species, C_high, C_low in [
        (
            "Sr",
            Sr_high,
            Sr_low
        ),
        (
            "Ba",
            Ba_high,
            Ba_low
        )
    ]:


        # Reference profiles for each XAn value.
        for XAn_value in XAn_values_fig5:

            I_ref, C_ref = (
                erf_like_reference_profile_fig5(
                    species,
                    XAn_value,
                    C_high,
                    C_low
                )
            )

            fig5_data[
                species
            ][
                XAn_value
            ] = {
                "I_ref": I_ref,
                "C_ref": C_ref
            }


        # Common analytical envelope anchored at XAn = 0.50.
        I_ref_anchor, C_ref_anchor = (
            erf_like_reference_profile_fig5(
                species,
                XAn_anchor,
                C_high,
                C_low
            )
        )


        sigma_anchor, lower_anchor, upper_anchor = (
            build_max_envelope(
                C_ref_anchor,
                uncertainty_rel_frac_fig5,
                uncertainty_abs_ppm_fig5
            )
        )


        fig5_data[
            species
        ][
            "anchor"
        ] = {
            "XAn": XAn_anchor,
            "C_ref": C_ref_anchor,
            "sigma": sigma_anchor,
            "lower": lower_anchor,
            "upper": upper_anchor
        }


    # ============================================================
    # 9. HIDDEN RESIDENCE USING THE COMMON XAn = 0.50 ENVELOPE
    # ============================================================

    results_main = {
        "Sr": {},
        "Ba": {}
    }

    results_inset = {
        "Sr": {},
        "Ba": {}
    }


    for species in [
        "Sr",
        "Ba"
    ]:


        Cstart_common = (
            fig5_data[
                species
            ][
                "anchor"
            ][
                "C_ref"
            ]
        )

        lower_common = (
            fig5_data[
                species
            ][
                "anchor"
            ][
                "lower"
            ]
        )

        upper_common = (
            fig5_data[
                species
            ][
                "anchor"
            ][
                "upper"
            ]
        )


        for XAn_value in XAn_values_fig5:


            Cstart = Cstart_common


            # ------------------------------------------------------------
            # Common-envelope calculations
            # ------------------------------------------------------------

            tcrit_cool = np.zeros_like(
                T_cooling_fig5,
                dtype=float
            )

            tcrit_warm = np.zeros_like(
                T_warming_fig5,
                dtype=float
            )


            for k, Tres in enumerate(
                T_cooling_fig5
            ):

                tcrit_cool[k] = (
                    tcrit_bisection_fd(
                        Cstart,
                        Tres,
                        XAn_value,
                        species,
                        lower_common,
                        upper_common,
                        x_um,
                        t_high_init_yr=t_high_init_yr,
                        t_high_max_yr=t_high_max_yr,
                        tol_rel=tol_rel_time,
                        x_window=x_window_um,
                        alpha_safety=alpha_safety
                    )
                )


            for k, Tres in enumerate(
                T_warming_fig5
            ):

                tcrit_warm[k] = (
                    tcrit_bisection_fd(
                        Cstart,
                        Tres,
                        XAn_value,
                        species,
                        lower_common,
                        upper_common,
                        x_um,
                        t_high_init_yr=t_high_init_yr,
                        t_high_max_yr=t_high_max_yr,
                        tol_rel=tol_rel_time,
                        x_window=x_window_um,
                        alpha_safety=alpha_safety
                    )
                )


            results_main[
                species
            ][
                XAn_value
            ] = {
                "cool": tcrit_cool,
                "warm": tcrit_warm
            }


            # ------------------------------------------------------------
            # Self-envelope calculations
            # ------------------------------------------------------------

            sigma_self, lower_self, upper_self = (
                build_max_envelope(
                    Cstart,
                    uncertainty_rel_frac_fig5,
                    uncertainty_abs_ppm_fig5
                )
            )


            tcrit_cool_self = np.zeros_like(
                T_cooling_fig5,
                dtype=float
            )

            tcrit_warm_self = np.zeros_like(
                T_warming_fig5,
                dtype=float
            )


            for k, Tres in enumerate(
                T_cooling_fig5
            ):

                tcrit_cool_self[k] = (
                    tcrit_bisection_fd(
                        Cstart,
                        Tres,
                        XAn_value,
                        species,
                        lower_self,
                        upper_self,
                        x_um,
                        t_high_init_yr=t_high_init_yr,
                        t_high_max_yr=t_high_max_yr,
                        tol_rel=tol_rel_time,
                        x_window=x_window_um,
                        alpha_safety=alpha_safety
                    )
                )


            for k, Tres in enumerate(
                T_warming_fig5
            ):

                tcrit_warm_self[k] = (
                    tcrit_bisection_fd(
                        Cstart,
                        Tres,
                        XAn_value,
                        species,
                        lower_self,
                        upper_self,
                        x_um,
                        t_high_init_yr=t_high_init_yr,
                        t_high_max_yr=t_high_max_yr,
                        tol_rel=tol_rel_time,
                        x_window=x_window_um,
                        alpha_safety=alpha_safety
                    )
                )


            results_inset[
                species
            ][
                XAn_value
            ] = {
                "cool": tcrit_cool_self,
                "warm": tcrit_warm_self
            }


    # ============================================================
    # 10. HEATING CASE — HIDDEN PREHISTORY
    # ============================================================

    # Hidden residence occurs below the initial temperature of the
    # heating event and is followed by the 900–1000 °C heating path.
    XAn_plot_heating_prehistory = [
        0.00,
        0.50,
        1.00
    ]


    T_hidden_before_heating_fig5 = np.linspace(
        Tinitial_ref2_fig5
        - deltaT_res_fig5,
        Tinitial_ref2_fig5,
        n_Tres_fig5
    )


    results_heating_prehistory = {
        "Sr": {},
        "Ba": {}
    }


    total_heat_pre_models = (
        2
        * len(
            XAn_plot_heating_prehistory
        )
        * len(
            T_hidden_before_heating_fig5
        )
    )


    pbar_heat_pre = tqdm(
        total=total_heat_pre_models,
        desc="Figure 5 heating prehistory",
        unit="model"
    )


    for species in [
        "Sr",
        "Ba"
    ]:


        C_high = (
            Sr_high
            if species == "Sr"
            else Ba_high
        )

        C_low = (
            Sr_low
            if species == "Sr"
            else Ba_low
        )


        _, C_ref_heat_anchor = (
            erf_like_profile_custom_path_fig5(
                species,
                XAn_anchor,
                C_high,
                C_low,
                Tinitial_ref2_fig5,
                Tfinal_ref2_fig5,
                t_ref2_s_fig5
            )
        )


        sigma_heat, lower_heat, upper_heat = (
            build_max_envelope(
                C_ref_heat_anchor,
                uncertainty_rel_frac_fig5,
                uncertainty_abs_ppm_fig5
            )
        )


        for XAn_value in (
            XAn_plot_heating_prehistory
        ):


            tcrit_heat_pre = np.zeros_like(
                T_hidden_before_heating_fig5,
                dtype=float
            )


            for k, Tres in enumerate(
                T_hidden_before_heating_fig5
            ):


                tcrit_heat_pre[k] = (
                    tcrit_hidden_erf_then_path_fd(
                        Tres,
                        XAn_value,
                        species,
                        C_high,
                        C_low,
                        lower_heat,
                        upper_heat,
                        Tinitial_ref2_fig5,
                        Tfinal_ref2_fig5,
                        t_ref2_s_fig5,
                        x_um,
                        t_high_init_yr=t_high_init_yr,
                        t_high_max_yr=t_high_max_yr,
                        tol_rel=tol_rel_time,
                        x_window=x_window_um
                    )
                )


                pbar_heat_pre.update(1)


            results_heating_prehistory[
                species
            ][
                XAn_value
            ] = tcrit_heat_pre


    pbar_heat_pre.close()


    # ============================================================
    # 11. CREATE FIGURE 5
    # ============================================================

    fig5, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 10.0)
    )


    axSrProf, axBaProf = (
        axes[0, 0],
        axes[0, 1]
    )

    axSrCool, axBaCool = (
        axes[1, 0],
        axes[1, 1]
    )

    # ============================================================
    # 12. HIDDEN-RESIDENCE TEMPERATURE ARRAYS
    # ============================================================

    dT_hidden_window_fig5 = 150.0


    # ============================================================
    # 13. PANEL A — COOLING REFERENCE PROFILES
    # ============================================================

    # Sr is plotted against the left y-axis and Ba against the
    # corresponding right y-axis.
    axProf = axSrProf

    axProf_Ba = (
        axProf.twinx()
    )


    axBaProf.set_visible(
        True
    )


    # Only XAn = 0.00, 0.50 and 1.00 are displayed.
    XAn_plot_profiles = [
        0.00,
        0.50,
        1.00
    ]


    color_Sr = "blue"
    color_Ba = "red"


    # XAn = 0.50 is emphasised in the plotted profiles.
    lw_map_profile = {
        0.00: 2.0,
        0.50: 3.2,
        1.00: 2.0
    }

    alpha_map_profile = {
        0.00: 0.55,
        0.50: 1.00,
        1.00: 0.55
    }

    ls_map_profile = {
        0.00: "--",
        0.50: "-",
        1.00: ":"
    }


    # Initial Sr profile.
    axProf.plot(
        x_um,
        Sr_initial,
        color="black",
        linewidth=2.4,
        linestyle="--",
        label="Initial profile"
    )


    # ------------------------------------------------------------
    # Sr reference profiles
    # ------------------------------------------------------------

    for XAn_value in XAn_plot_profiles:


        sr_ref = (
            fig5_data[
                "Sr"
            ][
                XAn_value
            ][
                "C_ref"
            ]
        )


        label_xan = (
            r"$X_{An}=0.50$"
            if abs(
                XAn_value - 0.50
            ) < 1e-9
            else rf"$X_{{An}}\approx{XAn_value:.2f}$"
        )


        axProf.plot(
            x_um,
            sr_ref,
            color=color_Sr,
            linewidth=lw_map_profile[
                XAn_value
            ],
            alpha=alpha_map_profile[
                XAn_value
            ],
            linestyle=ls_map_profile[
                XAn_value
            ],
            label=f"Sr {label_xan}",
            zorder=(
                4
                if abs(
                    XAn_value - 0.50
                ) < 1e-9
                else 2
            )
        )


    # ------------------------------------------------------------
    # Ba reference profiles
    # ------------------------------------------------------------

    for XAn_value in XAn_plot_profiles:


        ba_ref = (
            fig5_data[
                "Ba"
            ][
                XAn_value
            ][
                "C_ref"
            ]
        )


        label_xan = (
            r"$X_{An}=0.50$"
            if abs(
                XAn_value - 0.50
            ) < 1e-9
            else rf"$X_{{An}}\approx{XAn_value:.2f}$"
        )


        axProf_Ba.plot(
            x_um,
            ba_ref,
            color=color_Ba,
            linewidth=lw_map_profile[
                XAn_value
            ],
            alpha=alpha_map_profile[
                XAn_value
            ],
            linestyle=ls_map_profile[
                XAn_value
            ],
            label=f"Ba {label_xan}",
            zorder=(
                4
                if abs(
                    XAn_value - 0.50
                ) < 1e-9
                else 2
            )
        )


    axProf.set_xlim(
        *xlim_profiles_fig5
    )


    # These limits align Sr 900 ppm with Ba 300 ppm and
    # Sr 400 ppm with Ba 100 ppm.
    axProf.set_ylim(
        360,
        940
    )

    axProf_Ba.set_ylim(
        84,
        316
    )


    axProf.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    axProf.set_ylabel(
        "Sr (ppm)",
        color=color_Sr,
        fontsize=LABEL_FS
    )

    axProf_Ba.set_ylabel(
        "Ba (ppm)",
        color=color_Ba,
        fontsize=LABEL_FS
    )


    axProf.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axProf.tick_params(
        axis="y",
        labelcolor=color_Sr
    )

    axProf_Ba.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axProf_Ba.tick_params(
        axis="y",
        labelcolor=color_Ba
    )


    axProf.set_title(
        rf"COOLING: "
        rf"$T_i$={Tinitial_ref_fig5:.0f}°C, "
        rf"$T_f$={Tfinal_ref_fig5:.0f}°C, "
        rf"{t_years:.0f} yr",
        fontsize=TITLE_FS
    )


    axProf.grid(
        alpha=0.25
    )


    handles_Sr, labels_Sr = (
        axProf.get_legend_handles_labels()
    )

    handles_Ba, labels_Ba = (
        axProf_Ba.get_legend_handles_labels()
    )


    axProf.text(
        0.98,
        0.98,
        "a",
        transform=axProf.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    axProf.legend(
        handles_Sr + handles_Ba,
        labels_Sr + labels_Ba,
        ncol=1,
        fontsize=LEGEND_FS,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 14. PANEL C — HEATING REFERENCE PROFILES
    # ============================================================

    axProf2 = axBaProf

    axProf2_Ba = (
        axProf2.twinx()
    )


    # Initial Sr profile.
    axProf2.plot(
        x_um,
        Sr_initial,
        color="black",
        linewidth=2.4,
        linestyle="--",
        label="Initial profile"
    )

    # ------------------------------------------------------------
    # Sr reference profiles
    # ------------------------------------------------------------

    for XAn_value in XAn_plot_profiles:


        _, sr_ref2 = (
            erf_like_profile_custom_path_fig5(
                "Sr",
                XAn_value,
                Sr_high,
                Sr_low,
                Tinitial_ref2_fig5,
                Tfinal_ref2_fig5,
                t_ref2_s_fig5
            )
        )


        label_xan = (
            r"$X_{An}=0.50$"
            if abs(
                XAn_value - 0.50
            ) < 1e-9
            else rf"$X_{{An}}\approx{XAn_value:.2f}$"
        )


        axProf2.plot(
            x_um,
            sr_ref2,
            color=color_Sr,
            linewidth=lw_map_profile[
                XAn_value
            ],
            alpha=alpha_map_profile[
                XAn_value
            ],
            linestyle=ls_map_profile[
                XAn_value
            ],
            label=f"Sr {label_xan}",
            zorder=(
                4
                if abs(
                    XAn_value - 0.50
                ) < 1e-9
                else 2
            )
        )


    # ------------------------------------------------------------
    # Ba reference profiles
    # ------------------------------------------------------------

    for XAn_value in XAn_plot_profiles:


        _, ba_ref2 = (
            erf_like_profile_custom_path_fig5(
                "Ba",
                XAn_value,
                Ba_high,
                Ba_low,
                Tinitial_ref2_fig5,
                Tfinal_ref2_fig5,
                t_ref2_s_fig5
            )
        )


        label_xan = (
            r"$X_{An}=0.50$"
            if abs(
                XAn_value - 0.50
            ) < 1e-9
            else rf"$X_{{An}}\approx{XAn_value:.2f}$"
        )


        axProf2_Ba.plot(
            x_um,
            ba_ref2,
            color=color_Ba,
            linewidth=lw_map_profile[
                XAn_value
            ],
            alpha=alpha_map_profile[
                XAn_value
            ],
            linestyle=ls_map_profile[
                XAn_value
            ],
            label=f"Ba {label_xan}",
            zorder=(
                4
                if abs(
                    XAn_value - 0.50
                ) < 1e-9
                else 2
            )
        )


    axProf2.set_xlim(
        *xlim_profiles_fig5
    )


    axProf2.set_ylim(
        360,
        940
    )

    axProf2_Ba.set_ylim(
        84,
        316
    )


    axProf2.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    axProf2.set_ylabel(
        "Sr (ppm)",
        color=color_Sr,
        fontsize=LABEL_FS
    )

    axProf2_Ba.set_ylabel(
        "Ba (ppm)",
        color=color_Ba,
        fontsize=LABEL_FS
    )


    axProf2.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axProf2.tick_params(
        axis="y",
        labelcolor=color_Sr
    )

    axProf2_Ba.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    axProf2_Ba.tick_params(
        axis="y",
        labelcolor=color_Ba
    )


    axProf2.set_title(
        rf"HEATING: "
        rf"$T_i$={Tinitial_ref2_fig5:.0f}°C, "
        rf"$T_f$={Tfinal_ref2_fig5:.0f}°C, "
        rf"{t_ref2_yr_fig5:.0f} yr",
        fontsize=TITLE_FS
    )


    axProf2.grid(
        alpha=0.25
    )


    axProf2.text(
        0.98,
        0.98,
        "c",
        transform=axProf2.transAxes,
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        fontweight="bold"
    )


    handles_Sr2, labels_Sr2 = (
        axProf2.get_legend_handles_labels()
    )

    handles_Ba2, labels_Ba2 = (
        axProf2_Ba.get_legend_handles_labels()
    )


    axProf2.legend(
        handles_Sr2 + handles_Ba2,
        labels_Sr2 + labels_Ba2,
        ncol=1,
        fontsize=LEGEND_FS,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 15. HIDDEN-RESIDENCE PLOTTING PARAMETERS
    # ============================================================

    lw_map = {
        0.00: 2.0,
        0.50: 3.2,
        1.00: 2.0
    }

    alpha_map = {
        0.00: 0.55,
        0.50: 1.00,
        1.00: 0.55
    }

    ls_map = {
        0.00: "--",
        0.50: "-",
        1.00: ":"
    }


    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox


    # ============================================================
    # 16. SINGLE-SPECIES HIDDEN-TIME PLOTTING FUNCTION
    # ============================================================

    def plot_main_panel(
        ax_main: Axes,
        T_array,
        species,
        branch_key,
        color_main
    ):


        XAn_plot_main = [
            0.00,
            0.50,
            1.00
        ]


        for XAn_value in XAn_plot_main:


            y = (
                results_main[
                    species
                ][
                    XAn_value
                ][
                    branch_key
                ]
            )


            ax_main.plot(
                T_array,
                y,
                color=color_main,
                linewidth=lw_map[
                    XAn_value
                ],
                alpha=alpha_map[
                    XAn_value
                ],
                linestyle=ls_map[
                    XAn_value
                ],
                label=(
                    f"XAn={XAn_value:.2f}"
                    if XAn_value
                    != XAn_anchor
                    else "XAn=0.50 (anchor)"
                )
            )


        ax_main.set_yscale(
            "log"
        )

        ax_main.grid(
            alpha=0.25
        )

        ax_main.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FS
        )

        ax_main.legend(
            fontsize=LEGEND_FS,
            frameon=True,
            ncol=3,
            loc="lower left"
        )


        yfit = (
            results_main[
                species
            ][
                XAn_anchor
            ][
                branch_key
            ].copy()
        )


        mask = (
            yfit > 0
        )


        if np.count_nonzero(
            mask
        ) >= 2:


            T_fit = T_array[
                mask
            ]

            L_fit = np.log(
                yfit[
                    mask
                ]
            )


            b, loga = np.polyfit(
                T_fit,
                L_fit,
                1
            )

            a = np.exp(
                loga
            )


            ax_main.text(
                0.04,
                0.06,
                (
                    f"XAn=0.50: "
                    f"y = {a:.2e} · "
                    f"exp({b:.3e}·T)"
                ),
                ha="left",
                va="bottom",
                fontsize=LEGEND_FS,
                transform=ax_main.transAxes,
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.85,
                    boxstyle="round,pad=0.25"
                )
            )


    # ============================================================
    # 17. PANEL B — COOLING HIDDEN RESIDENCE
    # ============================================================

    def plot_combined_hidden_time_panel(
        ax,
        T_array,
        branch_key,
        panel_title
    ):


        XAn_plot_main = [
            0.00,
            0.50,
            1.00
        ]


        # Sr curves.
        for XAn_value in XAn_plot_main:


            y_sr = (
                results_main[
                    "Sr"
                ][
                    XAn_value
                ][
                    branch_key
                ]
            )


            ax.plot(
                T_array,
                y_sr,
                color=color_Sr,
                linewidth=lw_map[
                    XAn_value
                ],
                alpha=alpha_map[
                    XAn_value
                ],
                linestyle=ls_map[
                    XAn_value
                ],
                label=(
                    rf"Sr "
                    rf"$X_{{An}}="
                    rf"{XAn_value:.2f}$"
                )
            )


        # Ba curves.
        for XAn_value in XAn_plot_main:


            y_ba = (
                results_main[
                    "Ba"
                ][
                    XAn_value
                ][
                    branch_key
                ]
            )


            ax.plot(
                T_array,
                y_ba,
                color=color_Ba,
                linewidth=lw_map[
                    XAn_value
                ],
                alpha=alpha_map[
                    XAn_value
                ],
                linestyle=ls_map[
                    XAn_value
                ],
                label=(
                    rf"Ba "
                    rf"$X_{{An}}="
                    rf"{XAn_value:.2f}$"
                )
            )


        ax.set_yscale(
            "log"
        )


        ax.set_xlabel(
            "Temperature (°C)",
            fontsize=LABEL_FS
        )

        ax.set_ylabel(
            "Hidden residence time (yr)",
            fontsize=LABEL_FS
        )


        ax.set_title(
            panel_title,
            fontsize=TITLE_FS
        )


        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FS
        )


        ax.grid(
            alpha=0.25
        )


        ax.text(
            0.98,
            0.98,
            "b",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TITLE_FS,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="white",
                pad=0.25
            )
        )


        ax.legend(
            fontsize=LEGEND_FS,
            frameon=True,
            ncol=1,
            loc="lower left"
        )


    # ============================================================
    # 18. PANEL D — PRE-HEATING HIDDEN RESIDENCE
    # ============================================================

    def plot_heating_prehistory_panel(
        ax
    ):


        XAn_plot_main = [
            0.00,
            0.50,
            1.00
        ]


        for XAn_value in XAn_plot_main:


            ax.plot(
                T_hidden_before_heating_fig5,
                results_heating_prehistory[
                    "Sr"
                ][
                    XAn_value
                ],
                color=color_Sr,
                linewidth=lw_map[
                    XAn_value
                ],
                alpha=alpha_map[
                    XAn_value
                ],
                linestyle=ls_map[
                    XAn_value
                ],
                label=(
                    rf"Sr "
                    rf"$X_{{An}}="
                    rf"{XAn_value:.2f}$"
                )
            )


            ax.plot(
                T_hidden_before_heating_fig5,
                results_heating_prehistory[
                    "Ba"
                ][
                    XAn_value
                ],
                color=color_Ba,
                linewidth=lw_map[
                    XAn_value
                ],
                alpha=alpha_map[
                    XAn_value
                ],
                linestyle=ls_map[
                    XAn_value
                ],
                label=(
                    rf"Ba "
                    rf"$X_{{An}}="
                    rf"{XAn_value:.2f}$"
                )
            )


        ax.axvline(
            Tinitial_ref2_fig5,
            color="black",
            linestyle="--",
            linewidth=1.2
        )


        ax.set_yscale(
            "log"
        )


        ax.set_xlabel(
            (
                r"Pre-heating residence temperature "
                r"$T_{residence}$ (°C)"
            ),
            fontsize=LABEL_FS
        )


        ax.set_ylabel(
            (
                r"Hidden residence "
                r"$t_{hidden}$ (yr)"
            ),
            fontsize=LABEL_FS
        )


        ax.set_title(
            (
                "HEATING: Hidden "
                "pre-heating residence time"
            ),
            fontsize=TITLE_FS
        )


        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FS
        )


        ax.grid(
            alpha=0.25
        )


        ax.legend(
            fontsize=LEGEND_FS,
            frameon=True,
            ncol=1,
            loc="lower left"
        )


        ax.text(
            0.98,
            0.98,
            "d",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TITLE_FS,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="white",
                pad=0.25
            )
        )


    # ============================================================
    # 19. PLOT HIDDEN-RESIDENCE PANELS
    # ============================================================

    plot_combined_hidden_time_panel(
        axSrCool,
        T_cooling_fig5,
        "cool",
        "COOLING: Hidden residence time"
    )


    plot_heating_prehistory_panel(
        axBaCool
    )

    # ============================================================
    # 20. FINAL FIGURE FORMATTING
    # ============================================================

    for ax in [
        axSrCool,
        axBaCool
    ]:

        ax.set_ylabel(
            (
                r"Hidden residence "
                r"$t_{hidden}$ (yr)"
            ),
            fontsize=LABEL_FS
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FS
        )


    axSrCool.set_title(
        (
            "COOLING: Hidden residence "
            "time after cooling"
        ),
        fontsize=TITLE_FS
    )


    axBaCool.set_title(
        (
            "HEATING: Hidden residence "
            "time before heating"
        ),
        fontsize=TITLE_FS
    )

    axSrCool.set_xlabel(
        (
            r"Diffusion temperature "
            r"$T_{residence}$ (°C)"
        ),
        fontsize=LABEL_FS
    )


    axBaCool.set_xlabel(
        (
            r"Diffusion temperature "
            r"$T_{residence}$ (°C)"
        ),
        fontsize=LABEL_FS
    )


    for ax in [
        axSrCool,
        axBaCool
    ]:

        ax.axvline(
            Tfinal_ref_fig5,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.7
        )


    # Use the same y-axis range for panels b and d.
    shared_ylim_hidden = (
        min(
            axSrCool.get_ylim()[0],
            axBaCool.get_ylim()[0]
        ),
        max(
            axSrCool.get_ylim()[1],
            axBaCool.get_ylim()[1]
        )
    )


    axSrCool.set_ylim(
        shared_ylim_hidden
    )

    axBaCool.set_ylim(
        shared_ylim_hidden
    )



    # ============================================================
    # 21. DISPLAY FIGURE
    # ============================================================

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )

    plt.show()


    # ============================================================
    # 22. SAVE FIGURE
    # ============================================================

    output_name_fig5 = (
        f"Figure_5_hidden_residence_time_"
        f"{selected_method_fig5}"
    )

    output_file_fig5 = (
        OUTPUT_DIR
        / f"{output_name_fig5}.png"
    )

    fig5.savefig(
        output_file_fig5,
        dpi=300,
        bbox_inches="tight"
    )

# ============================================================
# END OF MANUSCRIPT FIGURE 5
# ============================================================
#endregion












#region MANUSCRIPT FIGURE 6 — TIME DEPENDENCY OF DIFFUSION PROFILES AND HIDDEN RESIDENCE
# ============================================================
# MANUSCRIPT FIGURE 6 — TIME DEPENDENCY OF DIFFUSION PROFILES AND HIDDEN RESIDENCE
# ============================================================

# Manuscript Figure 6 investigates how the duration of a thermal event affects:
#
#   1. the final Sr and Ba diffusion profiles, and
#   2. the amount of additional residence time that could remain
#      analytically undetected.
#
# Two thermal histories are considered:
#
#   Cooling: 1000 → 900 °C
#   Heating:  900 → 1100 °C
# These can be modified in the "MODEL PARAMETERS" section below. 
#
# For each case, diffusion profiles are calculated for several event
# durations. The resulting profiles are then used as references for
# calculating how much additional diffusion could occur before the
# profile moves outside the selected analytical uncertainty envelope.
#
# For the cooling case, hidden residence occurs AFTER the thermal event.
# For the heating case, hidden residence occurs BEFORE the heating event.

# ============================================================
# 1. FIGURE SETTINGS
# ============================================================

# Figure 6 font sizes.
TITLE_FS = 14
LABEL_FS = 13
TICK_FS = 12
LEGEND_FS = 11

# Figure control.
run_fig6 = True
use_EPMA_fig6 = False  # If False, SIMS uncertainty is used.


if run_fig6:


    # ============================================================
    # 2. ANALYTICAL UNCERTAINTY
    # ============================================================

    selected_method_fig6 = (
        "EPMA"
        if use_EPMA_fig6
        else "SIMS"
    )

    selected_cfg_fig6 = uncertainty_models[
        selected_method_fig6
    ]

    uncertainty_rel_frac_fig6 = (
        selected_cfg_fig6["rel_frac"]
    )

    uncertainty_abs_ppm_fig6 = (
        selected_cfg_fig6["abs_ppm"]
    )



    # ============================================================
    # 3. MODEL PARAMETERS
    # ============================================================

    # Plagioclase composition.
    XAn_fig6 = 0.50


    # Cooling thermal path.
    Ti_cool_fig6 = 1000.0
    Tf_cool_fig6 = 900.0


    # Heating thermal path.
    Ti_heat_fig6 = 900.0
    Tf_heat_fig6 = 1100.0


    # Residence temperatures evaluated after cooling.
    T_hidden_cooling_fig6 = np.linspace(
        Tf_cool_fig6,
        Tf_cool_fig6 - 150.0,
        8
    )


    # Residence temperatures evaluated before heating.
    T_hidden_heating_fig6 = np.linspace(
        Ti_heat_fig6 - 150.0,
        Ti_heat_fig6,
        8
    )

    # Detectability window around the diffusion interface.
    x_window_fig6 = 80.0


    # Bisection parameters.
    t_high_init_yr_fig6 = 10.0
    t_high_max_yr_fig6 = 1e8
    tol_rel_fig6 = 0.05


    # ============================================================
    # 4. EVENT-DURATION CASES
    # ============================================================

    # Individual cases can be included or excluded from the
    # calculations and figure using the "plot" switch.
    time_cases_fig6 = {

        "1 day": {
            "duration_yr": 1.0 / 365.25,
            "plot": False,
            "color": "purple",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.85,
        },

        "1 month": {
            "duration_yr": 1.0 / 12.0,
            "plot": False,
            "color": "blue",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.85,
        },

        "1 year": {
            "duration_yr": 1.0,
            "plot": True,
            "color": "green",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.85,
        },

        "10 years": {
            "duration_yr": 10.0,
            "plot": False,
            "color": "green",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.85,
        },

        "100 years": {
            "duration_yr": 100.0,
            "plot": True,
            "color": "orange",
            "linestyle": "-",
            "linewidth": 2.3,
            "alpha": 0.90,
        },

        "1000 years": {
            "duration_yr": 1000.0,
            "plot": True,
            "color": "purple",
            "linestyle": "-",
            "linewidth": 2.5,
            "alpha": 0.95,
        },
    }


    active_time_cases_fig6 = {
        key: cfg
        for key, cfg in time_cases_fig6.items()
        if cfg["plot"]
    }


    # ============================================================
    # 5. ANALYTICAL ENVELOPE FUNCTION
    # ============================================================

    def build_analytical_envelope_fig6(
        profile
    ):

        sigma = np.maximum(
            uncertainty_rel_frac_fig6 * profile,
            uncertainty_abs_ppm_fig6
        )

        return (
            sigma,
            profile - sigma,
            profile + sigma
        )


    # ============================================================
    # 6. LINEAR THERMAL-PATH PROFILE
    # ============================================================

    def erf_profile_for_linear_path_fig6(
        species,
        XAn_value,
        C_high,
        C_low,
        Ti_C,
        Tf_C,
        duration_yr,
        n_steps=800
    ):

        duration_s = (
            duration_yr
            * seconds_per_year
        )


        t_series_local = np.linspace(
            0.0,
            duration_s,
            n_steps
        )


        T_series_C_local = (
            Ti_C
            + (Tf_C - Ti_C)
            * (
                t_series_local
                / duration_s
            )
        )


        T_series_K_local = (
            T_series_C_local
            + 273.15
        )


        D_series_local = diffusion_coefficient(
            T_series_K_local,
            XAn_value,
            species
        )


        I_local = np.trapz(
            D_series_local,
            t_series_local
        )


        profile_local = erf_profile_from_integral(
            x_um,
            I_local,
            C_high,
            C_low,
            x0_um
        )


        return (
            I_local,
            profile_local
        )


    # ============================================================
    # 7. ENVELOPE ACCEPTANCE TEST
    # ============================================================

    def inside_envelope_window_fig6(
        C,
        lower,
        upper,
        x_um,
        x_window=None
    ):

        if x_window is None:

            m = slice(None)

        else:

            m = (
                (x_um >= -x_window)
                & (x_um <= x_window)
            )


        return np.all(
            (C[m] >= lower[m])
            & (C[m] <= upper[m])
        )


    # ============================================================
    # 8. CONSTANT-TEMPERATURE FD EVOLUTION
    # ============================================================

    def evolve_fd_constT_to_time_fig6(
        C0,
        t_target_s,
        D_const,
        x_um,
        alpha_safety=0.4
    ):

        if (
            D_const <= 0.0
            or t_target_s <= 0.0
        ):

            return (
                C0.astype(float).copy()
            )


        x_m_local = (
            x_um
            * 1e-6
        )

        dx_local = (
            x_m_local[1]
            - x_m_local[0]
        )


        dt_stable = (
            alpha_safety
            * dx_local**2
            / D_const
        )


        n_steps = max(
            1,
            int(
                np.ceil(
                    t_target_s
                    / dt_stable
                )
            )
        )


        dt_eff = (
            t_target_s
            / n_steps
        )


        alpha = (
            D_const
            * dt_eff
            / dx_local**2
        )


        C = (
            C0.astype(float).copy()
        )


        for _ in range(n_steps):

            Cn = C.copy()


            C[1:-1] = (
                Cn[1:-1]
                + alpha
                * (
                    Cn[2:]
                    - 2.0 * Cn[1:-1]
                    + Cn[:-2]
                )
            )


            C[0] = C[1]
            C[-1] = C[-2]


        return C


    # ============================================================
    # 9. HIDDEN RESIDENCE AFTER A DIFFUSION PROFILE
    # ============================================================

    def tcrit_after_profile_fd_fig6(
        Cstart,
        Tres_C,
        XAn_value,
        species,
        lower,
        upper,
        x_um,
        t_high_init_yr=10.0,
        t_high_max_yr=1e6,
        tol_rel=0.01,
        x_window=None
    ):

        D_const = diffusion_coefficient(
            Tres_C + 273.15,
            XAn_value,
            species
        )


        if D_const <= 0.0:

            return 0.0


        t_low_s = 0.0

        t_high_s = (
            t_high_init_yr
            * seconds_per_year
        )


        # ------------------------------------------------------------
        # Bracket the critical residence time
        # ------------------------------------------------------------

        while True:


            C_test = evolve_fd_constT_to_time_fig6(
                Cstart,
                t_high_s,
                D_const,
                x_um
            )


            if not inside_envelope_window_fig6(
                C_test,
                lower,
                upper,
                x_um,
                x_window
            ):

                break


            t_low_s = t_high_s
            t_high_s *= 2.0


            if (
                t_high_s
                > t_high_max_yr
                * seconds_per_year
            ):

                return (
                    t_high_s
                    / seconds_per_year
                )


        # ------------------------------------------------------------
        # Bisection search
        # ------------------------------------------------------------

        while (
            t_high_s - t_low_s
        ) > (
            tol_rel
            * max(t_high_s, 1.0)
        ):


            t_mid_s = (
                0.5
                * (
                    t_low_s
                    + t_high_s
                )
            )


            C_mid = evolve_fd_constT_to_time_fig6(
                Cstart,
                t_mid_s,
                D_const,
                x_um
            )


            if inside_envelope_window_fig6(
                C_mid,
                lower,
                upper,
                x_um,
                x_window
            ):

                t_low_s = t_mid_s

            else:

                t_high_s = t_mid_s


        return (
            t_low_s
            / seconds_per_year
        )


    # ============================================================
    # 10. HIDDEN RESIDENCE BEFORE A HEATING PATH
    # ============================================================

    def tcrit_hidden_then_path_fd_fig6(
        Tres_C,
        XAn_value,
        species,
        C_high,
        C_low,
        lower,
        upper,
        Ti_path_C,
        Tf_path_C,
        t_path_s,
        x_um,
        t_high_init_yr=10.0,
        t_high_max_yr=1e6,
        tol_rel=0.01,
        x_window=None
    ):
        """
        Hidden residence occurs first at Tres_C and is followed
        by the prescribed heating path.
        """

        D_hidden = diffusion_coefficient(
            Tres_C + 273.15,
            XAn_value,
            species
        )


        if D_hidden <= 0.0:

            return 0.0


        XAn_profile_local = np.full_like(
            x_um,
            XAn_value,
            dtype=float
        )


        x_m_local = (
            x_um
            * 1e-6
        )

        dx_local = (
            x_m_local[1]
            - x_m_local[0]
        )


        # Determine the largest diffusivity along the heating path.
        T_test_K = np.linspace(
            Ti_path_C + 273.15,
            Tf_path_C + 273.15,
            500
        )


        D_test = diffusion_coefficient(
            T_test_K,
            XAn_value,
            species
        )


        D_max_path = np.max(
            D_test
        )


        n_steps_path = int(
            np.ceil(
                D_max_path
                * t_path_s
                / (
                    0.40
                    * dx_local**2
                )
            )
        ) + 2


        t_series_path = np.linspace(
            0.0,
            t_path_s,
            n_steps_path
        )


        T_series_path_C = (
            Ti_path_C
            + (
                Tf_path_C
                - Ti_path_C
            )
            * (
                t_series_path
                / t_path_s
            )
        )


        T_series_path_K = (
            T_series_path_C
            + 273.15
        )


        def final_profile_after_hidden(
            t_hidden_s
        ):

            I_hidden = (
                D_hidden
                * t_hidden_s
            )


            C_hidden = erf_profile_from_integral(
                x_um,
                I_hidden,
                C_high,
                C_low,
                x0_um
            )


            C_final = diffuse_1D_variable_XAn(
                T_series_path_K,
                t_path_s,
                x_um,
                C_hidden,
                XAn_profile_local,
                species
            )


            return C_final


        # ------------------------------------------------------------
        # Bracket the critical hidden residence time
        # ------------------------------------------------------------

        t_low_s = 0.0

        t_high_s = (
            t_high_init_yr
            * seconds_per_year
        )


        while True:


            C_test = (
                final_profile_after_hidden(
                    t_high_s
                )
            )


            if not inside_envelope_window_fig6(
                C_test,
                lower,
                upper,
                x_um,
                x_window
            ):

                break


            t_low_s = t_high_s
            t_high_s *= 2.0


            if (
                t_high_s
                > t_high_max_yr
                * seconds_per_year
            ):

                return (
                    t_high_s
                    / seconds_per_year
                )


        # ------------------------------------------------------------
        # Bisection search
        # ------------------------------------------------------------

        while (
            t_high_s - t_low_s
        ) > (
            tol_rel
            * max(t_high_s, 1.0)
        ):


            t_mid_s = (
                0.5
                * (
                    t_low_s
                    + t_high_s
                )
            )


            C_mid = (
                final_profile_after_hidden(
                    t_mid_s
                )
            )


            if inside_envelope_window_fig6(
                C_mid,
                lower,
                upper,
                x_um,
                x_window
            ):

                t_low_s = t_mid_s

            else:

                t_high_s = t_mid_s


        return (
            t_low_s
            / seconds_per_year
        )


    # ============================================================
    # 11. CALCULATE DIFFUSION PROFILES
    # ============================================================

    fig6_profiles = {

        "cooling": {
            "Sr": {},
            "Ba": {}
        },

        "heating": {
            "Sr": {},
            "Ba": {}
        },
    }


    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        duration_yr = cfg[
            "duration_yr"
        ]


        for species, C_high, C_low in [

            (
                "Sr",
                Sr_high,
                Sr_low
            ),

            (
                "Ba",
                Ba_high,
                Ba_low
            )
        ]:


            _, C_cool = (
                erf_profile_for_linear_path_fig6(
                    species,
                    XAn_fig6,
                    C_high,
                    C_low,
                    Ti_cool_fig6,
                    Tf_cool_fig6,
                    duration_yr
                )
            )


            _, C_heat = (
                erf_profile_for_linear_path_fig6(
                    species,
                    XAn_fig6,
                    C_high,
                    C_low,
                    Ti_heat_fig6,
                    Tf_heat_fig6,
                    duration_yr
                )
            )


            fig6_profiles[
                "cooling"
            ][
                species
            ][
                time_label
            ] = C_cool


            fig6_profiles[
                "heating"
            ][
                species
            ][
                time_label
            ] = C_heat


    # ============================================================
    # 12. HIDDEN RESIDENCE CALCULATIONS
    # ============================================================

    fig6_hidden = {

        "cooling": {
            "Sr": {},
            "Ba": {}
        },

        "heating": {
            "Sr": {},
            "Ba": {}
        },
    }


    # ============================================================
    # 13. COOLING — HIDDEN RESIDENCE AFTER THE EVENT
    # ============================================================

    total_cooling_models_fig6 = (
        len(
            active_time_cases_fig6
        )
        * 2
        * len(
            T_hidden_cooling_fig6
        )
    )


    pbar_cool_fig6 = tqdm(
        total=total_cooling_models_fig6,
        desc="Figure 6 cooling hidden residence",
        unit="model"
    )


    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        for species in [
            "Sr",
            "Ba"
        ]:


            C_reference = (
                fig6_profiles[
                    "cooling"
                ][
                    species
                ][
                    time_label
                ]
            )


            _, lower_ref, upper_ref = (
                build_analytical_envelope_fig6(
                    C_reference
                )
            )


            tcrit_array = np.zeros_like(
                T_hidden_cooling_fig6,
                dtype=float
            )


            for k, Tres in enumerate(
                T_hidden_cooling_fig6
            ):


                tcrit_array[k] = (
                    tcrit_after_profile_fd_fig6(
                        C_reference,
                        Tres,
                        XAn_fig6,
                        species,
                        lower_ref,
                        upper_ref,
                        x_um,
                        t_high_init_yr=t_high_init_yr_fig6,
                        t_high_max_yr=t_high_max_yr_fig6,
                        tol_rel=tol_rel_fig6,
                        x_window=x_window_fig6
                    )
                )


                pbar_cool_fig6.update(
                    1
                )


            fig6_hidden[
                "cooling"
            ][
                species
            ][
                time_label
            ] = tcrit_array


    pbar_cool_fig6.close()


    # ============================================================
    # 14. HEATING — HIDDEN RESIDENCE BEFORE THE EVENT
    # ============================================================

    total_heating_models_fig6 = (
        len(
            active_time_cases_fig6
        )
        * 2
        * len(
            T_hidden_heating_fig6
        )
    )


    pbar_heat_fig6 = tqdm(
        total=total_heating_models_fig6,
        desc="Figure 6 heating hidden residence",
        unit="model"
    )


    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        duration_yr = cfg[
            "duration_yr"
        ]

        duration_s = (
            duration_yr
            * seconds_per_year
        )


        for species, C_high, C_low in [

            (
                "Sr",
                Sr_high,
                Sr_low
            ),

            (
                "Ba",
                Ba_high,
                Ba_low
            )
        ]:


            C_reference = (
                fig6_profiles[
                    "heating"
                ][
                    species
                ][
                    time_label
                ]
            )


            _, lower_ref, upper_ref = (
                build_analytical_envelope_fig6(
                    C_reference
                )
            )


            tcrit_array = np.zeros_like(
                T_hidden_heating_fig6,
                dtype=float
            )


            for k, Tres in enumerate(
                T_hidden_heating_fig6
            ):


                tcrit_array[k] = (
                    tcrit_hidden_then_path_fd_fig6(
                        Tres,
                        XAn_fig6,
                        species,
                        C_high,
                        C_low,
                        lower_ref,
                        upper_ref,
                        Ti_heat_fig6,
                        Tf_heat_fig6,
                        duration_s,
                        x_um,
                        t_high_init_yr=t_high_init_yr_fig6,
                        t_high_max_yr=t_high_max_yr_fig6,
                        tol_rel=tol_rel_fig6,
                        x_window=x_window_fig6
                    )
                )


                pbar_heat_fig6.update(
                    1
                )


            fig6_hidden[
                "heating"
            ][
                species
            ][
                time_label
            ] = tcrit_array


    pbar_heat_fig6.close()


    # ============================================================
    # 15. CREATE FIGURE 6
    # ============================================================

    fig6, axes7 = plt.subplots(
        2,
        2,
        figsize=(12.5, 10.0)
    )


    ax7a, ax7b = (
        axes7[0, 0],
        axes7[0, 1]
    )

    ax7c, ax7d = (
        axes7[1, 0],
        axes7[1, 1]
    )


    color_Sr_fig6 = "blue"
    color_Ba_fig6 = "red"


    # ============================================================
    # 16. PANEL A — COOLING PROFILES
    # ============================================================

    ax7a_Ba = (
        ax7a.twinx()
    )


    ax7a.plot(
        x_um,
        Sr_initial,
        color="black",
        linestyle="-",
        linewidth=2.2,
        label="Initial profile"
    )


    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        ax7a.plot(
            x_um,
            fig6_profiles[
                "cooling"
            ][
                "Sr"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Sr {time_label}"
        )


        ax7a_Ba.plot(
            x_um,
            fig6_profiles[
                "cooling"
            ][
                "Ba"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="--",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Ba {time_label}"
        )


    ax7a.set_xlim(
        -200,
        200
    )

    ax7a.set_ylim(
        360,
        940
    )

    ax7a_Ba.set_ylim(
        84,
        316
    )


    ax7a.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    ax7a.set_ylabel(
        "Sr (ppm)",
        color=color_Sr_fig6,
        fontsize=LABEL_FS
    )

    ax7a_Ba.set_ylabel(
        "Ba (ppm)",
        color=color_Ba_fig6,
        fontsize=LABEL_FS
    )


    ax7a.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    ax7a.tick_params(
        axis="y",
        labelcolor=color_Sr_fig6
    )

    ax7a_Ba.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    ax7a_Ba.tick_params(
        axis="y",
        labelcolor=color_Ba_fig6
    )


    ax7a.set_title(
        rf"COOLING: "
        rf"$T_i$={Ti_cool_fig6:.0f} °C, "
        rf"$T_f$={Tf_cool_fig6:.0f} °C, "
        rf"$X_{{An}}$={XAn_fig6:.2f}",
        fontsize=TITLE_FS
    )


    ax7a.grid(
        alpha=0.25
    )


    handles_a1, labels_a1 = (
        ax7a.get_legend_handles_labels()
    )

    handles_a2, labels_a2 = (
        ax7a_Ba.get_legend_handles_labels()
    )


    ax7a.legend(
        handles_a1 + handles_a2,
        labels_a1 + labels_a2,
        fontsize=LEGEND_FS,
        ncol=1,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 17. PANEL B — HEATING PROFILES
    # ============================================================

    ax7b_Ba = (
        ax7b.twinx()
    )


    ax7b.plot(
        x_um,
        Sr_initial,
        color="black",
        linestyle="-",
        linewidth=2.2,
        label="Initial profile"
    )


    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        ax7b.plot(
            x_um,
            fig6_profiles[
                "heating"
            ][
                "Sr"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Sr {time_label}"
        )


        ax7b_Ba.plot(
            x_um,
            fig6_profiles[
                "heating"
            ][
                "Ba"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="--",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Ba {time_label}"
        )


    ax7b.set_xlim(
        -200,
        200
    )

    ax7b.set_ylim(
        360,
        940
    )

    ax7b_Ba.set_ylim(
        84,
        316
    )


    ax7b.set_xlabel(
        "Distance (µm)",
        fontsize=LABEL_FS
    )

    ax7b.set_ylabel(
        "Sr (ppm)",
        color=color_Sr_fig6,
        fontsize=LABEL_FS
    )

    ax7b_Ba.set_ylabel(
        "Ba (ppm)",
        color=color_Ba_fig6,
        fontsize=LABEL_FS
    )


    ax7b.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    ax7b.tick_params(
        axis="y",
        labelcolor=color_Sr_fig6
    )

    ax7b_Ba.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )

    ax7b_Ba.tick_params(
        axis="y",
        labelcolor=color_Ba_fig6
    )


    ax7b.set_title(
        rf"HEATING: "
        rf"$T_i$={Ti_heat_fig6:.0f} °C, "
        rf"$T_f$={Tf_heat_fig6:.0f} °C, "
        rf"$X_{{An}}$={XAn_fig6:.2f}",
        fontsize=TITLE_FS
    )


    ax7b.grid(
        alpha=0.25
    )


    handles_b1, labels_b1 = (
        ax7b.get_legend_handles_labels()
    )

    handles_b2, labels_b2 = (
        ax7b_Ba.get_legend_handles_labels()
    )


    ax7b.legend(
        handles_b1 + handles_b2,
        labels_b1 + labels_b2,
        fontsize=LEGEND_FS,
        ncol=1,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 18. PANEL C — HIDDEN RESIDENCE AFTER COOLING
    # ============================================================

    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        ax7c.plot(
            T_hidden_cooling_fig6,
            fig6_hidden[
                "cooling"
            ][
                "Sr"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="-",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Sr {time_label}"
        )


        ax7c.plot(
            T_hidden_cooling_fig6,
            fig6_hidden[
                "cooling"
            ][
                "Ba"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="--",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Ba {time_label}"
        )


    ax7c.axvline(
        Tf_cool_fig6,
        color="black",
        linestyle="--",
        linewidth=1.2
    )


    ax7c.set_yscale(
        "log"
    )


    ax7c.set_xlabel(
        r"Diffusion temperature $T_{residence}$ (°C)",
        fontsize=LABEL_FS
    )

    ax7c.set_ylabel(
        r"Hidden residence $t_{hidden}$ (yr)",
        fontsize=LABEL_FS
    )


    ax7c.set_title(
        "COOLING: Hidden residence after cooling",
        fontsize=TITLE_FS
    )


    ax7c.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )


    ax7c.grid(
        alpha=0.25
    )


    ax7c.legend(
        fontsize=LEGEND_FS,
        ncol=1,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 19. PANEL D — HIDDEN RESIDENCE BEFORE HEATING
    # ============================================================

    for time_label, cfg in (
        active_time_cases_fig6.items()
    ):


        ax7d.plot(
            T_hidden_heating_fig6,
            fig6_hidden[
                "heating"
            ][
                "Sr"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="-",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Sr {time_label}"
        )


        ax7d.plot(
            T_hidden_heating_fig6,
            fig6_hidden[
                "heating"
            ][
                "Ba"
            ][
                time_label
            ],
            color=cfg["color"],
            linestyle="--",
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"Ba {time_label}"
        )


    ax7d.axvline(
        Ti_heat_fig6,
        color="black",
        linestyle="--",
        linewidth=1.2
    )


    ax7d.set_yscale(
        "log"
    )


    ax7d.set_xlabel(
        r"Pre-heating residence temperature $T_{residence}$ (°C)",
        fontsize=LABEL_FS
    )

    ax7d.set_ylabel(
        r"Hidden residence $t_{hidden}$ (yr)",
        fontsize=LABEL_FS
    )


    ax7d.set_title(
        "HEATING: Hidden residence before heating",
        fontsize=TITLE_FS
    )


    ax7d.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )


    ax7d.grid(
        alpha=0.25
    )


    ax7d.legend(
        fontsize=LEGEND_FS,
        ncol=1,
        frameon=True,
        loc="lower left"
    )


    # ============================================================
    # 20. MATCH HIDDEN-RESIDENCE Y-AXES
    # ============================================================

    # Panels c and d use the same logarithmic y-axis range.
    shared_ylim_hidden = (
        min(
            ax7c.get_ylim()[0],
            ax7d.get_ylim()[0]
        ),
        max(
            ax7c.get_ylim()[1],
            ax7d.get_ylim()[1]
        )
    )


    ax7c.set_ylim(
        shared_ylim_hidden
    )

    ax7d.set_ylim(
        shared_ylim_hidden
    )


    # ============================================================
    # 21. DISPLAY FIGURE
    # ============================================================

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )

    plt.show()


    # ============================================================
    # 22. SAVE FIGURE
    # ============================================================

    output_name_fig6 = (
        f"Figure_6_time_dependency_profiles_hidden_residence_"
        f"{selected_method_fig6}"
    )


    output_file_fig6 = (
        OUTPUT_DIR
        / f"{output_name_fig6}.png"
    )


    fig6.savefig(
        output_file_fig6,
        dpi=300,
        bbox_inches="tight"
    )


# ============================================================
# END OF MANUSCRIPT FIGURE 6
# ============================================================
#endregion
















#region MANUSCRIPT FIGURE 7 — RECOVERY OF ISOTHERMAL TEMPERATURE AND DURATION
from scipy.special import erfinv
# ============================================================
# MANUSCRIPT FIGURE 7 — RECOVERY OF ISOTHERMAL TEMPERATURE AND DURATION
# USING PAIRED Sr–Ba DIFFUSION WIDTHS
# ============================================================

# Manuscript Figure 7 illustrates how paired Sr and Ba diffusion profiles can
# be used to recover an effective diffusion temperature and duration.
#
# For diffusion from the same interface, Sr and Ba experience the
# same temperature and duration. Their different diffusivities
# therefore produce different diffusion widths.
#
# For an isothermal diffusion profile:
#
#       L^2 ∝ D t
#
# and consequently:
#
#       L_Ba^2 / L_Sr^2 = D_Ba / D_Sr
#
# because the common diffusion duration cancels from the ratio.
#
# The measured squared-width ratio can therefore be compared with
# the temperature-dependent diffusivity ratio D_Ba / D_Sr to recover
# temperature. Once temperature is known, the duration can be
# calculated independently from either Sr or Ba.
#
# Both finite-difference and analytical profiles are calculated here
# so that the recovery can be compared between the two approaches.


# ============================================================
# 1. FIGURE CONTROL
# ============================================================

run_fig7 = True


if run_fig7:


    # ============================================================
    # 2. REFERENCE EXPERIMENT
    # ============================================================

    # Reference isothermal conditions used to generate the synthetic Sr and Ba diffusion profiles.
    T_ref_fig7 = 1000.0       # °C
    t_ref_yr_fig7 = 300.0     # years
    XAn_fig7 = 0.50


    # Convert the reference conditions to the units required by the diffusion calculations.
    t_ref_s_fig7 = (
        t_ref_yr_fig7
        * seconds_per_year
    )

    T_ref_K_fig7 = (
        T_ref_fig7
        + 273.15
    )


    # ============================================================
    # 3. DIFFUSION-WIDTH DEFINITION
    # ============================================================

    # Diffusion widths are measured between 1% and 99% of the original concentration step.
    width_frac_fig7 = 0.01

    # For an error-function profile, the 1–99% width can be written:
    #
    #       L = k * sqrt(D t)
    #
    # where k depends only on the concentration fractions selected for measuring the width.
    k_width_fig7 = 4.0 * erfinv(
        1.0
        - 2.0 * width_frac_fig7
    )

    # Squaring the width relation gives:
    #
    #       L^2 = k^2 D t
    #
    # This factor is later used to recover the diffusion duration.
    width_factor_fig7 = (
        k_width_fig7**2
    )


    # ============================================================
    # 4. GENERATE FINITE-DIFFERENCE REFERENCE PROFILES
    # ============================================================

    # Figure 7 assumes a spatially constant anorthite fraction.
    XAn_profile_fig7 = np.full_like(
        x_um,
        XAn_fig7,
        dtype=float
    )


    # Initial sharp Sr concentration step.
    Sr_initial_fig7 = step_initial_profile(
        x_um,
        Sr_high,
        Sr_low,
        x0_um
    )


    # Initial sharp Ba concentration step.
    Ba_initial_fig7 = step_initial_profile(
        x_um,
        Ba_high,
        Ba_low,
        x0_um
    )


    # ------------------------------------------------------------
    # Determine a stable finite-difference timestep
    # ------------------------------------------------------------

    x_m_fig7 = (
        x_um
        * 1e-6
    )

    dx_fig7 = (
        x_m_fig7[1]
        - x_m_fig7[0]
    )


    # Sr and Ba diffusivities at the reference temperature.
    D_Sr_ref_fig7 = diffusion_coefficient(
        T_ref_K_fig7,
        XAn_fig7,
        "Sr"
    )


    D_Ba_ref_fig7 = diffusion_coefficient(
        T_ref_K_fig7,
        XAn_fig7,
        "Ba"
    )


    # The largest diffusivity controls the stability requirement.
    D_max_fig7 = max(
        D_Sr_ref_fig7,
        D_Ba_ref_fig7
    )


    dt_max_fig7 = (
        0.40
        * dx_fig7**2
        / D_max_fig7
    )


    # The temperature array contains n_steps points and therefore
    # n_steps - 1 finite-difference time intervals.
    n_steps_fig7 = (
        int(
            np.ceil(
                t_ref_s_fig7
                / dt_max_fig7
            )
        )
        + 1
    )


    # Constant-temperature history for the isothermal experiment.
    T_series_fig7 = np.full(
        n_steps_fig7,
        T_ref_K_fig7
    )


    # ------------------------------------------------------------
    # Sr finite-difference profile
    # ------------------------------------------------------------

    Sr_profile_fig7 = diffuse_1D_variable_XAn(
        T_series_fig7,
        t_ref_s_fig7,
        x_um,
        Sr_initial_fig7,
        XAn_profile_fig7,
        "Sr"
    )


    # ------------------------------------------------------------
    # Ba finite-difference profile
    # ------------------------------------------------------------

    Ba_profile_fig7 = diffuse_1D_variable_XAn(
        T_series_fig7,
        t_ref_s_fig7,
        x_um,
        Ba_initial_fig7,
        XAn_profile_fig7,
        "Ba"
    )


    # ============================================================
    # 5. ANALYTICAL ERROR-FUNCTION PROFILES
    # ============================================================

    # For constant temperature, integrated diffusivity is simply:
    #
    #       I = D t
    #
    # Analytical error-function profiles are calculated using the
    # same reference temperature and duration as the FD models.
    I_Sr_fig7 = (
        D_Sr_ref_fig7
        * t_ref_s_fig7
    )

    I_Ba_fig7 = (
        D_Ba_ref_fig7
        * t_ref_s_fig7
    )


    Sr_profile_erf_fig7 = erf_profile_from_integral(
        x_um,
        I_Sr_fig7,
        Sr_high,
        Sr_low,
        x0_um
    )


    Ba_profile_erf_fig7 = erf_profile_from_integral(
        x_um,
        I_Ba_fig7,
        Ba_high,
        Ba_low,
        x0_um
    )


    # ============================================================
    # 6. DIFFUSION-WIDTH MEASUREMENT
    # ============================================================

    def diffusion_width_fig7(
        profile,
        distance_um,
        C_high,
        C_low,
        frac=0.01
    ):
        """
        Measure the distance between the 1% and 99% concentration
        levels of a monotonically decreasing diffusion profile.

        Linear interpolation is used to determine the positions at
        which the profile crosses the two concentration thresholds.
        """

        concentration_range = (
            C_high
            - C_low
        )


        # Concentration corresponding to 99% of the original step.
        C_at_99 = (
            C_high
            - frac
            * concentration_range
        )


        # Concentration corresponding to 1% of the original step.
        C_at_01 = (
            C_low
            + frac
            * concentration_range
        )


        # The concentration profile decreases with distance.
        # Arrays are reversed because np.interp requires an increasing interpolation coordinate.
        x_at_99 = np.interp(
            C_at_99,
            profile[::-1],
            distance_um[::-1]
        )


        x_at_01 = np.interp(
            C_at_01,
            profile[::-1],
            distance_um[::-1]
        )


        return abs(
            x_at_01
            - x_at_99
        )


    # ============================================================
    # 7. MEASURE Sr AND Ba PROFILE WIDTHS
    # ============================================================

    # ------------------------------------------------------------
    # Finite-difference widths
    # ------------------------------------------------------------

    width_Sr_fig7 = diffusion_width_fig7(
        Sr_profile_fig7,
        x_um,
        Sr_high,
        Sr_low,
        width_frac_fig7
    )


    width_Ba_fig7 = diffusion_width_fig7(
        Ba_profile_fig7,
        x_um,
        Ba_high,
        Ba_low,
        width_frac_fig7
    )


    # Squared-width ratio measured from the FD profiles.
    width_ratio_fig7 = (
        width_Ba_fig7**2
        / width_Sr_fig7**2
    )


    # ------------------------------------------------------------
    # Analytical-profile widths
    # ------------------------------------------------------------

    width_Sr_erf_fig7 = diffusion_width_fig7(
        Sr_profile_erf_fig7,
        x_um,
        Sr_high,
        Sr_low,
        width_frac_fig7
    )


    width_Ba_erf_fig7 = diffusion_width_fig7(
        Ba_profile_erf_fig7,
        x_um,
        Ba_high,
        Ba_low,
        width_frac_fig7
    )


    width_ratio_erf_fig7 = (
        width_Ba_erf_fig7**2
        / width_Sr_erf_fig7**2
    )


    # ============================================================
    # 8. DIFFUSIVITY-RATIO CURVE AS A FUNCTION OF TEMPERATURE
    # ============================================================

    # The Sr and Ba diffusivities are evaluated over a broad
    # temperature range. Their ratio provides the calibration curve
    # against which the measured squared-width ratio is compared.
    T_range_fig7 = np.linspace(
        700.0,
        1200.0,
        2001
    )


    D_Sr_range_fig7 = diffusion_coefficient(
        T_range_fig7 + 273.15,
        XAn_fig7,
        "Sr"
    )


    D_Ba_range_fig7 = diffusion_coefficient(
        T_range_fig7 + 273.15,
        XAn_fig7,
        "Ba"
    )


    D_ratio_fig7 = (
        D_Ba_range_fig7
        / D_Sr_range_fig7
    )


    # ============================================================
    # 9. RECOVER EFFECTIVE TEMPERATURE
    # ============================================================

    # D_Ba / D_Sr decreases with increasing temperature.
    #
    # The measured squared-width ratio is therefore interpolated
    # against the diffusivity-ratio curve to recover the temperature
    # at which:
    #
    #       D_Ba / D_Sr = L_Ba^2 / L_Sr^2
    #
    # Arrays are reversed so that np.interp receives an increasing
    # interpolation coordinate.

    T_solution_fig7 = np.interp(
        width_ratio_fig7,
        D_ratio_fig7[::-1],
        T_range_fig7[::-1]
    )


    # Equivalent recovery using widths from the analytical profiles.
    T_solution_erf_fig7 = np.interp(
        width_ratio_erf_fig7,
        D_ratio_fig7[::-1],
        T_range_fig7[::-1]
    )


    # ============================================================
    # 10. RECOVER DURATION FROM Sr AND Ba
    # ============================================================

    # Once temperature is known, the corresponding Sr and Ba
    # diffusivities can be calculated and the diffusion duration
    # recovered independently from each profile:
    #
    #       t = L^2 / (k^2 D)

    D_Sr_solution_fig7 = diffusion_coefficient(
        T_solution_fig7 + 273.15,
        XAn_fig7,
        "Sr"
    )


    D_Ba_solution_fig7 = diffusion_coefficient(
        T_solution_fig7 + 273.15,
        XAn_fig7,
        "Ba"
    )


    # Convert measured widths from µm to metres.
    width_Sr_m_fig7 = (
        width_Sr_fig7
        * 1e-6
    )

    width_Ba_m_fig7 = (
        width_Ba_fig7
        * 1e-6
    )


    # Duration independently recovered from Sr.
    t_Sr_s_fig7 = (
        width_Sr_m_fig7**2
        / (
            width_factor_fig7
            * D_Sr_solution_fig7
        )
    )


    # Duration independently recovered from Ba.
    t_Ba_s_fig7 = (
        width_Ba_m_fig7**2
        / (
            width_factor_fig7
            * D_Ba_solution_fig7
        )
    )


    t_Sr_yr_fig7 = (
        t_Sr_s_fig7
        / seconds_per_year
    )


    t_Ba_yr_fig7 = (
        t_Ba_s_fig7
        / seconds_per_year
    )


    # ============================================================
    # 11. ANALYTICAL-PROFILE DURATION RECOVERY
    # ============================================================

    # Repeat the same duration calculation using the temperature
    # and widths recovered from the analytical profiles.

    D_Sr_solution_erf_fig7 = diffusion_coefficient(
        T_solution_erf_fig7 + 273.15,
        XAn_fig7,
        "Sr"
    )


    D_Ba_solution_erf_fig7 = diffusion_coefficient(
        T_solution_erf_fig7 + 273.15,
        XAn_fig7,
        "Ba"
    )


    t_Sr_erf_yr_fig7 = (
        (width_Sr_erf_fig7 * 1e-6)**2
        / (
            width_factor_fig7
            * D_Sr_solution_erf_fig7
        )
    ) / seconds_per_year


    t_Ba_erf_yr_fig7 = (
        (width_Ba_erf_fig7 * 1e-6)**2
        / (
            width_factor_fig7
            * D_Ba_solution_erf_fig7
        )
    ) / seconds_per_year


    # ============================================================
    # 12. PRINT RECOVERED CONDITIONS
    # ============================================================

    print(
        "\n[FIG7] Reference conditions"
    )


    print(
        f"[FIG7] T = {T_ref_fig7:.1f} °C; "
        f"t = {t_ref_yr_fig7:.1f} yr; "
        f"XAn = {XAn_fig7:.2f}"
    )


    print(
        f"[FIG7] FD width ratio = "
        f"{width_ratio_fig7:.5f}"
    )


    print(
        f"[FIG7] Recovered FD temperature = "
        f"{T_solution_fig7:.2f} °C"
    )


    print(
        f"[FIG7] Recovered FD times: "
        f"Sr = {t_Sr_yr_fig7:.2f} yr; "
        f"Ba = {t_Ba_yr_fig7:.2f} yr"
    )


    # ============================================================
    # 13. CREATE FIGURE 7
    # ============================================================

    # The figure compares the temperature-dependent diffusivity
    # ratio with the squared-width ratio measured from the synthetic
    # diffusion profiles.
    fig7, ax7 = plt.subplots(
        figsize=(8.5, 6.5)
    )


    # Diffusivity-ratio curve.
    ax7.plot(
        T_range_fig7,
        D_ratio_fig7,
        color="red",
        linewidth=2.5,
        label=(
            r"$D_{\mathrm{Ba}}/D_{\mathrm{Sr}}$ "
            r"(Grocolas et al., 2025)"
        )
    )


    # Horizontal line representing the width ratio measured from
    # the finite-difference profiles.
    ax7.axhline(
        width_ratio_fig7,
        color="black",
        linewidth=1.8,
        label=(
            r"$L_{\mathrm{Ba}}^2/L_{\mathrm{Sr}}^2$ "
            r"(from FD profiles)"
        )
    )


    # Vertical line marks the temperature at which the measured
    # width ratio intersects the diffusivity-ratio curve.
    ax7.axvline(
        T_solution_fig7,
        color="black",
        linewidth=1.8
    )


    # Finite-difference solution.
    ax7.scatter(
        T_solution_fig7,
        width_ratio_fig7,
        color="black",
        s=75,
        zorder=5,
        label="FD-derived solution"
    )


    # Analytical-profile solution.
    ax7.scatter(
        T_solution_erf_fig7,
        width_ratio_erf_fig7,
        color="grey",
        edgecolor="black",
        linewidth=0.4,
        s=150,
        alpha=0.75,
        zorder=4,
        label="Analytical solution"
    )


    # ============================================================
    # 14. RESULTS SUMMARY
    # ============================================================

    # Display the input conditions, measured widths, recovered
    # temperatures, and independently recovered Sr and Ba durations.
    results_text_fig7 = (
        "Reference model:\n"
        rf"$T_{{profile}}$ = {T_ref_fig7:.1f} °C" "\n"
        rf"$t_{{profile}}$ = {t_ref_yr_fig7:.1f} yr" "\n"
        rf"$X_{{An}}$ = {XAn_fig7:.2f}" "\n\n"

        "Measured widths:\n"
        rf"$L_{{Sr,FD}}$ = {width_Sr_fig7:.2f} µm" "\n"
        rf"$L_{{Ba,FD}}$ = {width_Ba_fig7:.2f} µm" "\n\n"

        "Squared-width ratios:\n"
        rf"FD: $L_{{Ba}}^2/L_{{Sr}}^2$ = "
        rf"{width_ratio_fig7:.3f}" "\n"
        rf"Analytical: $L_{{Ba}}^2/L_{{Sr}}^2$ = "
        rf"{width_ratio_erf_fig7:.3f}" "\n\n"

        "Recovered temperatures:\n"
        rf"$T_{{eff,FD}}$ = {T_solution_fig7:.1f} °C" "\n"
        rf"$T_{{eff,analytical}}$ = "
        rf"{T_solution_erf_fig7:.1f} °C" "\n\n"

        "Recovered durations:\n"
        rf"$t_{{Sr,FD}}$ = {t_Sr_yr_fig7:.1f} yr" "\n"
        rf"$t_{{Sr,analytical}}$ = "
        rf"{t_Sr_erf_yr_fig7:.1f} yr" "\n"
        rf"$t_{{Ba,FD}}$ = {t_Ba_yr_fig7:.1f} yr" "\n"
        rf"$t_{{Ba,analytical}}$ = "
        rf"{t_Ba_erf_yr_fig7:.1f} yr"
    )


    ax7.text(
        0.98,
        0.98,
        results_text_fig7,
        transform=ax7.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        linespacing=1.15,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="black",
            alpha=0.95
        )
    )


    # ============================================================
    # 15. AXES AND FIGURE FORMATTING
    # ============================================================

    ax7.set_xlabel(
        "Temperature (°C)",
        fontsize=LABEL_FS
    )


    ax7.set_ylabel(
        r"$D_{\mathrm{Ba}}/D_{\mathrm{Sr}}$",
        fontsize=LABEL_FS
    )


    ax7.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS
    )


    ax7.grid(
        alpha=0.25
    )


    ax7.legend(
        loc="lower left",
        fontsize=LEGEND_FS,
        framealpha=0.95
    )


    fig7.tight_layout()


    # ============================================================
    # 16. SAVE FIGURE
    # ============================================================

    output_name_fig7 = (
        "Figure_7_isothermal_Sr_Ba_ratio_recovery"
    )


    output_file_fig7 = (
        OUTPUT_DIR
        / f"{output_name_fig7}.png"
    )


    fig7.savefig(
        output_file_fig7,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# END OF MANUSCRIPT FIGURE 7
# ============================================================
#endregion















#region MANUSCRIPT FIGURE 8 — REDUCING NON-UNIQUENESS WITH PAIRED Sr–Ba CONSTRAINTS
# ============================================================
# MANUSCRIPT FIGURE 8 — REDUCING NON-UNIQUENESS WITH PAIRED Sr–Ba CONSTRAINTS
# ============================================================

# Manuscript Figure 8 evaluates how combining information from Sr and Ba
# diffusion profiles reduces the range of thermal histories that
# can reproduce a synthetic reference model.
#
# A non-isothermal reference profile is generated for:
#
#       1000 → 900 °C over 300 years
#
# Candidate models explore different initial temperatures, final
# temperatures, and durations.
#
# Four levels of acceptance are calculated:
#
#   1. Ba profile only
#   2. Sr profile only
#   3. Ba and Sr profiles simultaneously
#   4. Ba + Sr profiles plus their squared-width ratio
#
# The final figure compares the last two cases.
#
# The squared-width ratio provides an additional constraint because
# Sr and Ba diffuse for the same duration but have different
# temperature-dependent diffusivities:
#
#       L_Ba^2 / L_Sr^2
#
# This ratio can also be used to determine an equivalent isothermal
# temperature and corresponding Sr and Ba diffusion durations.


# ============================================================
# 1. FIGURE SETTINGS
# ============================================================

# Figure 8 font sizes.
TITLE_FS = 14
LABEL_FS = 13
TICK_FS = 12
LEGEND_FS = 10

# Figure control.
run_fig8 = True

# Select the analytical uncertainty model used for profile acceptance.
use_EPMA_fig8 = False   # True = EPMA, False = SIMS


if run_fig8:


    # ============================================================
    # 2. ANALYTICAL UNCERTAINTY
    # ============================================================

    selected_method_fig8 = (
        "EPMA"
        if use_EPMA_fig8
        else "SIMS"
    )

    selected_cfg_fig8 = uncertainty_models[
        selected_method_fig8
    ]

    uncertainty_rel_frac_fig8 = (
        selected_cfg_fig8["rel_frac"]
    )

    uncertainty_abs_ppm_fig8 = (
        selected_cfg_fig8["abs_ppm"]
    )


    print(
        f"[FIG8] Using {selected_cfg_fig8['label']} "
        f"analytical uncertainty envelope"
    )


    # ============================================================
    # 3. REFERENCE MODEL
    # ============================================================

    # Constant plagioclase composition.
    XAn_fig8 = 0.50


    # Non-isothermal reference thermal history.
    Ti_ref_fig8 = 1000.0
    Tf_ref_fig8 = 900.0

    t_ref_yr_fig8 = 300.0

    t_ref_s_fig8 = (
        t_ref_yr_fig8
        * seconds_per_year
    )


    # ============================================================
    # 4. MODEL SEARCH DOMAIN
    # ============================================================

    # Candidate initial temperatures.
    Ti_search_fig8 = np.linspace(
        800.0,
        1300.0,
        50
    )


    # Candidate final temperatures.
    Tf_search_fig8 = np.linspace(
        800.0,
        1300.0,
        50
    )


    # Candidate durations from 1 to 10,000 years.
    t_search_yr_fig8 = np.logspace(
        0,
        4,
        140
    )


    # ============================================================
    # 5. WIDTH-RATIO AND INTEGRATION SETTINGS
    # ============================================================

    # Candidate models must reproduce the reference squared-width ratio within ±5%.
    width_ratio_rel_tol_fig8 = 0.05   # 5 %


    # Number of points used to integrate diffusivity along each linear thermal history.
    n_steps_integral_fig8 = 400


    # ============================================================
    # 6. ANALYTICAL UNCERTAINTY ENVELOPE
    # ============================================================

    def build_analytical_envelope_fig8(profile):
        """
        Construct the analytical uncertainty envelope surrounding
        a concentration profile.

        At every spatial position, uncertainty is defined by the
        larger of the relative uncertainty and the absolute ppm
        uncertainty.
        """

        sigma = np.maximum(
            uncertainty_rel_frac_fig8 * profile,
            uncertainty_abs_ppm_fig8
        )

        return (
            sigma,
            profile - sigma,
            profile + sigma
        )


    # ============================================================
    # 7. PROFILE FOR A LINEAR THERMAL HISTORY
    # ============================================================

    def erf_profile_for_linear_path_fig8(
        species,
        XAn_value,
        C_high,
        C_low,
        Ti_C,
        Tf_C,
        duration_yr,
        n_steps=n_steps_integral_fig8
    ):
        """
        Calculate an analytical diffusion profile for a linear
        temperature history.

        Diffusivity is evaluated through the complete thermal path
        and integrated through time. The resulting integrated
        diffusivity is used in the error-function solution.
        """

        duration_s = (
            duration_yr
            * seconds_per_year
        )


        # Time discretisation of the thermal history.
        t_series_local = np.linspace(
            0.0,
            duration_s,
            n_steps
        )


        # Linear temperature evolution from Ti to Tf.
        T_series_C_local = (
            Ti_C
            + (Tf_C - Ti_C)
            * (
                t_series_local
                / duration_s
            )
        )


        T_series_K_local = (
            T_series_C_local
            + 273.15
        )


        # Diffusivity at every temperature along the path.
        D_series_local = diffusion_coefficient(
            T_series_K_local,
            XAn_value,
            species
        )


        # Integrated diffusivity of the complete thermal event.
        I_local = np.trapz(
            D_series_local,
            t_series_local
        )


        # Corresponding analytical concentration profile.
        profile_local = erf_profile_from_integral(
            x_um,
            I_local,
            C_high,
            C_low,
            x0_um
        )


        return (
            I_local,
            profile_local
        )


    # ============================================================
    # 8. PROFILE ACCEPTANCE TEST
    # ============================================================

    def profile_inside_envelope_fig8(
        profile,
        lower,
        upper
    ):
        """
        Test whether the complete model profile lies within the
        analytical uncertainty envelope.
        """

        return np.all(
            (profile >= lower)
            & (profile <= upper)
        )


    # ============================================================
    # 9. DIFFUSION-WIDTH MEASUREMENT
    # ============================================================

    def diffusion_width_fig8(
        profile,
        x_um,
        C_high,
        C_low,
        frac=0.01
    ):
        """
        Measure the width between 1% and 99% of the concentration
        step for a monotonically decreasing diffusion profile.
        """

        C_range = (
            C_high
            - C_low
        )

        # Concentration thresholds defining the width.
        C_left = (
            C_high
            - frac * C_range
        )

        C_right = (
            C_low
            + frac * C_range
        )

        # The concentration profile decreases with distance. Reverse the arrays because np.interp requires an increasing interpolation coordinate.
        x_left = np.interp(
            C_left,
            profile[::-1],
            x_um[::-1]
        )


        x_right = np.interp(
            C_right,
            profile[::-1],
            x_um[::-1]
        )


        return abs(
            x_right
            - x_left
        )


    # ============================================================
    # 10. PAIRED Sr–Ba WIDTH RATIO
    # ============================================================

    def width_ratio_fig8(
        Ba_profile,
        Sr_profile
    ):
        """
        Calculate the squared Ba-to-Sr diffusion-width ratio:

            L_Ba² / L_Sr²
        """

        w_ba = diffusion_width_fig8(
            Ba_profile,
            x_um,
            Ba_high,
            Ba_low,
            frac=0.01
        )


        w_sr = diffusion_width_fig8(
            Sr_profile,
            x_um,
            Sr_high,
            Sr_low,
            frac=0.01
        )


        return (
            w_ba**2
            / w_sr**2
        )


    # ============================================================
    # 11. GENERATE REFERENCE Sr AND Ba PROFILES
    # ============================================================

    # Sr reference profile.
    _, Sr_ref_fig8 = (
        erf_profile_for_linear_path_fig8(
            "Sr",
            XAn_fig8,
            Sr_high,
            Sr_low,
            Ti_ref_fig8,
            Tf_ref_fig8,
            t_ref_yr_fig8
        )
    )


    # Ba reference profile.
    _, Ba_ref_fig8 = (
        erf_profile_for_linear_path_fig8(
            "Ba",
            XAn_fig8,
            Ba_high,
            Ba_low,
            Ti_ref_fig8,
            Tf_ref_fig8,
            t_ref_yr_fig8
        )
    )


    # ============================================================
    # 12. REFERENCE ANALYTICAL ENVELOPES
    # ============================================================

    # Sr analytical uncertainty limits.
    _, Sr_lower_fig8, Sr_upper_fig8 = (
        build_analytical_envelope_fig8(
            Sr_ref_fig8
        )
    )


    # Ba analytical uncertainty limits.
    _, Ba_lower_fig8, Ba_upper_fig8 = (
        build_analytical_envelope_fig8(
            Ba_ref_fig8
        )
    )


    # Squared-width ratio of the paired reference profiles.
    ref_width_ratio_fig8 = width_ratio_fig8(
        Ba_ref_fig8,
        Sr_ref_fig8
    )


    # ============================================================
    # 13. EQUIVALENT ISOTHERMAL SOLUTION
    # ============================================================

    # The non-isothermal reference profiles can also be represented
    # by an equivalent isothermal temperature derived from their
    # paired Sr–Ba width ratio.

    width_frac_eff_fig8 = 0.01


    # ------------------------------------------------------------
    # Measure Sr and Ba reference-profile widths
    # ------------------------------------------------------------

    width_Sr_ref_fig8 = diffusion_width_fig8(
        Sr_ref_fig8,
        x_um,
        Sr_high,
        Sr_low,
        frac=width_frac_eff_fig8
    )


    width_Ba_ref_fig8 = diffusion_width_fig8(
        Ba_ref_fig8,
        x_um,
        Ba_high,
        Ba_low,
        frac=width_frac_eff_fig8
    )


    # ------------------------------------------------------------
    # Calculate the isothermal diffusivity-ratio curve
    # ------------------------------------------------------------

    # A finely sampled temperature range is used to determine where
    # the isothermal diffusivity ratio equals the reference-profile
    # squared-width ratio.
    T_eff_range_fig8 = np.linspace(
        700.0,
        1200.0,
        5001
    )


    D_Sr_eff_range_fig8 = diffusion_coefficient(
        T_eff_range_fig8 + 273.15,
        XAn_fig8,
        "Sr"
    )


    D_Ba_eff_range_fig8 = diffusion_coefficient(
        T_eff_range_fig8 + 273.15,
        XAn_fig8,
        "Ba"
    )


    D_ratio_eff_fig8 = (
        D_Ba_eff_range_fig8
        / D_Sr_eff_range_fig8
    )


    # ============================================================
    # 14. RECOVER EFFECTIVE TEMPERATURE
    # ============================================================

    # Determine the temperature at which the isothermal
    # diffusivity ratio reproduces the squared-width ratio of the
    # non-isothermal reference profiles.
    Teff_fig8 = np.interp(
        ref_width_ratio_fig8,
        D_ratio_eff_fig8[::-1],
        T_eff_range_fig8[::-1]
    )


    # Sr and Ba diffusivities at the recovered effective temperature.
    D_Sr_Teff_fig8 = diffusion_coefficient(
        Teff_fig8 + 273.15,
        XAn_fig8,
        "Sr"
    )


    D_Ba_Teff_fig8 = diffusion_coefficient(
        Teff_fig8 + 273.15,
        XAn_fig8,
        "Ba"
    )


    # ============================================================
    # 15. EQUIVALENT ISOTHERMAL DURATIONS
    # ============================================================

    # Width conversion factor for the 1–99% criterion:
    #
    #       L = k sqrt(Dt)
    #
    k_width_eff_fig8 = 4.0 * erfinv(
        1.0
        - 2.0 * width_frac_eff_fig8
    )


    # Recover an equivalent duration independently from the Sr reference-profile width.
    t_Sr_eff_s_fig8 = (
        (width_Sr_ref_fig8 * 1e-6)**2
        / (
            k_width_eff_fig8**2
            * D_Sr_Teff_fig8
        )
    )


    # Recover an equivalent duration independently from the Ba reference-profile width.
    t_Ba_eff_s_fig8 = (
        (width_Ba_ref_fig8 * 1e-6)**2
        / (
            k_width_eff_fig8**2
            * D_Ba_Teff_fig8
        )
    )


    t_Sr_eff_yr_fig8 = (
        t_Sr_eff_s_fig8
        / seconds_per_year
    )


    t_Ba_eff_yr_fig8 = (
        t_Ba_eff_s_fig8
        / seconds_per_year
    )

    # ============================================================
    # 16. WIDTH-RATIO ACCEPTANCE INTERVAL
    # ============================================================

    # The reference width ratio is expanded by the selected relative
    # tolerance to define the interval accepted in the model search.
    ratio_lower_fig8 = (
        ref_width_ratio_fig8
        * (
            1.0
            - width_ratio_rel_tol_fig8
        )
    )


    ratio_upper_fig8 = (
        ref_width_ratio_fig8
        * (
            1.0
            + width_ratio_rel_tol_fig8
        )
    )

    # ============================================================
    # 17. MODEL-ACCEPTANCE CATEGORIES
    # ============================================================

    # Keep separate records for models accepted using individual and combined constraints.
    accepted_fig8 = {
        "Ba only": [],
        "Sr only": [],
        "Ba + Sr": [],
        "Ba + Sr + width ratio": []
    }


    # Total number of candidate thermal histories.
    total_models_fig8 = (
        len(Ti_search_fig8)
        * len(Tf_search_fig8)
        * len(t_search_yr_fig8)
    )


    pbar_fig8 = tqdm(
        total=total_models_fig8,
        desc="Figure 8 paired-constraint search",
        unit="model"
    )


    # ============================================================
    # 18. SEARCH THERMAL-HISTORY SPACE
    # ============================================================

    # Every combination of initial temperature, final temperature and duration is evaluated.
    for Ti_model in Ti_search_fig8:

        for Tf_model in Tf_search_fig8:

            for t_model_yr in t_search_yr_fig8:


                # ------------------------------------------------------------
                # Calculate candidate Sr profile
                # ------------------------------------------------------------

                _, Sr_model = (
                    erf_profile_for_linear_path_fig8(
                        "Sr",
                        XAn_fig8,
                        Sr_high,
                        Sr_low,
                        Ti_model,
                        Tf_model,
                        t_model_yr
                    )
                )


                # ------------------------------------------------------------
                # Calculate candidate Ba profile
                # ------------------------------------------------------------

                _, Ba_model = (
                    erf_profile_for_linear_path_fig8(
                        "Ba",
                        XAn_fig8,
                        Ba_high,
                        Ba_low,
                        Ti_model,
                        Tf_model,
                        t_model_yr
                    )
                )


                # ------------------------------------------------------------
                # Test Sr profile against its analytical envelope
                # ------------------------------------------------------------

                accept_Sr = profile_inside_envelope_fig8(
                    Sr_model,
                    Sr_lower_fig8,
                    Sr_upper_fig8
                )


                # ------------------------------------------------------------
                # Test Ba profile against its analytical envelope
                # ------------------------------------------------------------

                accept_Ba = profile_inside_envelope_fig8(
                    Ba_model,
                    Ba_lower_fig8,
                    Ba_upper_fig8
                )


                # ------------------------------------------------------------
                # Calculate paired Sr–Ba width ratio
                # ------------------------------------------------------------

                model_width_ratio = width_ratio_fig8(
                    Ba_model,
                    Sr_model
                )


                # Test whether the model width ratio lies within the
                # accepted interval surrounding the reference value.
                accept_ratio = (
                    model_width_ratio
                    >= ratio_lower_fig8
                    and model_width_ratio
                    <= ratio_upper_fig8
                )


                # Store the thermal parameters and width ratio.
                record = {
                    "Ti": Ti_model,
                    "Tf": Tf_model,
                    "t": t_model_yr,
                    "ratio": model_width_ratio
                }


                # ------------------------------------------------------------
                # Store accepted models for each constraint combination
                # ------------------------------------------------------------

                if accept_Ba:

                    accepted_fig8[
                        "Ba only"
                    ].append(
                        record
                    )


                if accept_Sr:

                    accepted_fig8[
                        "Sr only"
                    ].append(
                        record
                    )


                if (
                    accept_Ba
                    and accept_Sr
                ):

                    accepted_fig8[
                        "Ba + Sr"
                    ].append(
                        record
                    )


                if (
                    accept_Ba
                    and accept_Sr
                    and accept_ratio
                ):

                    accepted_fig8[
                        "Ba + Sr + width ratio"
                    ].append(
                        record
                    )


                pbar_fig8.update(
                    1
                )


    pbar_fig8.close()


    # ============================================================
    # 19. PRINT ACCEPTANCE STATISTICS
    # ============================================================

    for key in accepted_fig8:

        n_acc = len(
            accepted_fig8[key]
        )

        print(
            f"[FIG8] {key}: "
            f"{n_acc}/{total_models_fig8} "
            f"({100*n_acc/total_models_fig8:.2f} %)"
        )


    # ============================================================
    # 20. FIGURE LAYOUT
    # ============================================================

    fig8 = plt.figure(
        figsize=(6.5, 12.5)
    )


    # Top panel.
    ax7c = fig8.add_axes(
        [0.14, 0.59, 0.82, 0.34]
    )


    # Bottom panel.
    ax7d = fig8.add_axes(
        [0.14, 0.18, 0.82, 0.34],
        sharex=ax7c,
        sharey=ax7c
    )


    # Dedicated colour-bar axis.
    cax_fig8 = fig8.add_axes(
        [0.18, 0.1, 0.74, 0.020]
    )


    panel_map_fig8 = [
        ("Ba + Sr", ax7c, "a"),
        ("Ba + Sr + width ratio", ax7d, "b"),
    ]


    # ============================================================
    # 21. COLOUR NORMALISATION
    # ============================================================

    norm_fig8 = mcolors.LogNorm(
        vmin=np.min(t_search_yr_fig8),
        vmax=np.max(t_search_yr_fig8)
    )


    last_scatter_fig8 = None


    # ============================================================
    # 22. PLOT ACCEPTED THERMAL HISTORIES
    # ============================================================

    for panel_name, ax, panel_letter in panel_map_fig8:


        records = accepted_fig8[
            panel_name
        ]


        if len(records) > 0:


            Ti_plot = np.array(
                [
                    r["Ti"]
                    for r in records
                ]
            )


            Tf_plot = np.array(
                [
                    r["Tf"]
                    for r in records
                ]
            )


            t_plot = np.array(
                [
                    r["t"]
                    for r in records
                ]
            )


            last_scatter_fig8 = ax.scatter(
                Ti_plot,
                Tf_plot,
                c=t_plot,
                cmap="viridis",
                norm=norm_fig8,
                s=34,
                edgecolor="black",
                linewidth=0.15,
                alpha=0.85,
                label="Accepted solutions"
            )


            Ti_min, Ti_max = (
                np.min(Ti_plot),
                np.max(Ti_plot)
            )

            Tf_min, Tf_max = (
                np.min(Tf_plot),
                np.max(Tf_plot)
            )

            t_min, t_max = (
                np.min(t_plot),
                np.max(t_plot)
            )


        else:


            Ti_min = Ti_max = np.nan
            Tf_min = Tf_max = np.nan
            t_min = t_max = np.nan


            ax.text(
                0.5,
                0.5,
                "No accepted models",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11
            )


        # ------------------------------------------------------------
        # Synthetic reference model
        # ------------------------------------------------------------

        ax.scatter(
            [Ti_ref_fig8],
            [Tf_ref_fig8],
            marker="*",
            s=230,
            color="red",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
            label="Synthetic model"
        )


        # ------------------------------------------------------------
        # Equivalent isothermal model
        # ------------------------------------------------------------

        ax.scatter(
            [Teff_fig8],
            [Teff_fig8],
            marker="*",
            s=230,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
            label=r"$T_{\mathrm{eff}}$"
        )


        # ------------------------------------------------------------
        # Isothermal Ti = Tf reference line
        # ------------------------------------------------------------

        ax.plot(
            [
                np.min(Ti_search_fig8),
                np.max(Ti_search_fig8)
            ],
            [
                np.min(Ti_search_fig8),
                np.max(Ti_search_fig8)
            ],
            color="grey",
            linestyle="--",
            linewidth=1.1,
            label=r"$T_i=T_f$"
        )


        ax.legend(
            loc="lower right",
            fontsize=LEGEND_FS,
            framealpha=0.95
        )


        # ============================================================
        # 23. ACCEPTED-RANGE TEXT
        # ============================================================

        n_acc = len(
            records
        )


        delta_t_Sr_eff_fig8 = (
            t_Sr_eff_yr_fig8
            - t_ref_yr_fig8
        )


        delta_t_Ba_eff_fig8 = (
            t_Ba_eff_yr_fig8
            - t_ref_yr_fig8
        )


        # Width-ratio-derived values are shown only in panel b.
        if panel_name == "Ba + Sr + width ratio":


            equivalent_text_fig8 = (
                "\n\nWidth ratio-derived values:\n"
                rf"$T_{{eff}}$ = {Teff_fig8:.1f} °C "
                rf"({Ti_ref_fig8:.0f}$\rightarrow$"
                rf"{Tf_ref_fig8:.0f} °C)"
                "\n"
                rf"$t_{{Sr,eff}}$ = "
                rf"{t_Sr_eff_yr_fig8:.1f} yr "
                rf"({t_ref_yr_fig8:.1f} yr)"
                "\n"
                rf"$t_{{Ba,eff}}$ = "
                rf"{t_Ba_eff_yr_fig8:.1f} yr "
                rf"({t_ref_yr_fig8:.1f} yr)"
            )


        else:


            equivalent_text_fig8 = ""


        text_fig8 = (
            f"Accepted: "
            f"{n_acc}/{total_models_fig8}"
            f"({100*n_acc/total_models_fig8:.2f} %)"
            "\n\n"
            "Accepted range:\n"
            rf"$T_i$ = {Ti_min:.0f}–{Ti_max:.0f} °C"
            "\n"
            rf"$T_f$ = {Tf_min:.0f}–{Tf_max:.0f} °C"
            "\n"
            rf"$t$ = {t_min:.1f}–{t_max:.0f} yr"
            f"{equivalent_text_fig8}"
        )


        ax.text(
            0.03,
            0.97,
            text_fig8,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=LEGEND_FS,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.90
            )
        )


        # ------------------------------------------------------------
        # Panel label
        # ------------------------------------------------------------

        ax.text(
            0.98,
            0.98,
            panel_letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=22,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="white",
                pad=0.25
            )
        )


        ax.set_title(
            panel_name,
            fontsize=TITLE_FS
        )


        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FS
        )


        ax.grid(
            alpha=0.25
        )


    # ============================================================
    # 24. AXES
    # ============================================================

    for ax in [
        ax7c,
        ax7d
    ]:

        ax.set_xlabel(
            r"$T_{initial}$ (°C)",
            fontsize=LABEL_FS
        )


    ax7c.set_ylabel(
        r"$T_{final}$ (°C)",
        fontsize=LABEL_FS
    )


    ax7d.set_ylabel(
        r"$T_{final}$ (°C)",
        fontsize=LABEL_FS
    )


    ax7c.set_xlim(
        np.min(Ti_search_fig8),
        np.max(Ti_search_fig8)
    )


    ax7c.set_ylim(
        np.min(Tf_search_fig8),
        np.max(Tf_search_fig8)
    )


    # ============================================================
    # 25. ACCEPTED-DURATION COLOUR BAR
    # ============================================================

    if last_scatter_fig8 is not None:


        cbar7 = fig8.colorbar(
            last_scatter_fig8,
            cax=cax_fig8,
            orientation="horizontal"
        )


        cbar7.set_label(
            "Accepted duration (yr)",
            fontsize=LABEL_FS
        )


        cbar7.ax.tick_params(
            labelsize=TICK_FS
        )


    # ============================================================
    # 26. DISPLAY FIGURE
    # ============================================================

    plt.show()


    # ============================================================
    # 27. SAVE FIGURE
    # ============================================================

    output_name_fig8 = (
        f"Figure_8_paired_Sr_Ba_constraints_"
        f"{selected_method_fig8}"
    )


    output_file_fig8 = (
        OUTPUT_DIR
        / f"{output_name_fig8}.png"
    )


    fig8.savefig(
        output_file_fig8,
        dpi=300,
        bbox_inches="tight"
    )


    # ============================================================
    # END OF MANUSCRIPT FIGURE 8
    # ============================================================
#endregion