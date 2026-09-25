"""
utils.py
"""

import numpy as np
from pathlib import Path

from ..dvr import DVR
from ..ec import ECSystem
from typing import Callable
from dataclasses import dataclass
import copy
# =====================================================================================================
#                                                SWEEP UTILS
# =======================================================================================================
# -------------------------------------------------------------------------------------------------------
# One scanned point and the observables that can be read off it
# ------------------------------------------------------------------------------------------------------
@dataclass
class ScanPoint:
    """
    Everything an observable function might need from one scan point.
    `resonance_dvr`, `resonance_energy`, `resonance_state` are the resonance-
    channel closest_to_resonance result. `bound_system`, `bound_energy`, `bound_state`
    are the matching bound-channel results, populated only if the scan was given a 
    bound_dvr + bound_energy (needed by obs_telem; None otherwise).

    `phi` and `L` are always populated (whichever one isn't being scanned is still whatever
    value the DVR system was built with).
    """
    resonance_dvr: DVR
    resonance_energy: complex
    resonance_state: np.ndarray

    phi: complex
    L: float

    bound_dvr: DVR | None = None
    bound_energy: complex | None = None
    bound_state: np.ndarray | None = None

def obs_energy(pt: ScanPoint) -> complex:
    """ Observable: the resonance energy itself. """
    return pt.resonance_energy

def obs_rr(pt: ScanPoint, rotate_rr: bool=True) -> complex:
    """
    Observable: (r^2) of the resonance state. 
    Default usage is `{"rr": obs_rr}`. For the rare non-default case, use
    `functools.partial(obs_rr, rotate_rr=False)`.
    """
    return pt.resonance_dvr.compute_rr(pt.resonance_state, rotate_rr=rotate_rr)

def obs_telem(pt: ScanPoint, operator: str="E1", phase_fixed: bool=True) -> complex:
    """ 
    Observable: (psi_res | operator | psi_bound). Requires the scan call
    to be given bound_dvr and bound_energy. Default usage is `{"telem": obs_telem}`.
    For the rare non-default case, use `functools.partial(obs_telem, operator=...)`.

    Note: the transition matrix element is determined under the c-product up to a sign. 
    Thus, the sign of the transition matrix element below should be "phase-fixed" according
    to some reference value. This function defaults to fixing the sign such that the real component is positive.
    """
    if pt.bound_dvr is None:
        raise ValueError(
            "obs_telem needs a bound state -- pass bound_dvr and "
            "bound_energy to scan_vs_phi / scan_vs_L"
        )
    if (not np.isclose(pt.bound_dvr.L, pt.L)) or (pt.resonance_dvr.n != pt.bound_dvr.n):
        raise ValueError(
            f"Bound state DVR and resonance DVR need to have the same system length and same number of mesh points. "
            f"Got ({pt.bound_dvr.L:.5f}, {pt.bound_dvr.n}) vs ({pt.L:.5f}, {pt.resonance_dvr.n})."
        )
    telem = DVR.compute_transition_mat_element(
        pt.L,
        pt.bound_state,
        pt.resonance_state,
        rotation_angle=pt.phi,
        operator=operator
    )
    if phase_fixed:
        telem *= np.sign(np.real(telem))
    
    return telem

# ------------------------------------------------------------------------------------------
# Scan engines: one diagonalization per point, many observables per point
# ------------------------------------------------------------------------------------------

def scan_vs_phi(
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis: np.ndarray,
    observables: dict[str, Callable[[ScanPoint], complex]],
    bound_dvr: DVR | None = None,
    bound_energy: complex | None = None,
) -> dict[str, np.ndarray]:
    """
    Scans phi, evaluating every observable in `observables` at each scan point. Also includes a "bound_energy"
    key that tracks the bound state energy over every phi in `phis`. `phis` can be complex or pure real. 
    
    Returns
    -------
    dict[str, np.ndarray]:
        Dictionary of results. Keys match those of `observables`. Array has shape (
    """    
    resonance_energy = complex(resonance_energy)
    phis = np.asarray(phis)
    
    if bound_dvr is not None:
        bound_energy = complex(bound_energy)
        if bound_dvr.L != resonance_dvr.L:
            raise ValueError(
                f"The bound system and resonance system need to have the same system length. "
                f"Got {bound_dvr.L} vs {resonance_dvr.L}."
            )
        if bound_dvr.n != resonance_dvr.n:
            raise ValueError(
                f"The bound system and resonance system need to have the same number of mesh points. "
                f"Got {bound_dvr.n} vs {resonance_dvr.n}."
            )
            
    results = {name: [] for name in observables}
    results["bound_energies"] = []
    for phi in phis:
        phi = complex(phi)
        resonance_dvr.reset_rotation_angle(phi)
        renergy, rstate = resonance_dvr.closest_to_resonance(resonance_energy)

        benergy = bstate = None
        if bound_dvr is not None:
            bound_dvr.reset_rotation_angle(phi)
            benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
            benergy, bstate = benergy[0], bstate[:, 0]
            results["bound_energies"].append(benergy)

        # copy.copy copies the dvr instance at it's current state (current rotation angle). This isn't strictly
        # necessary since we immediately compute observables right after the ScanPoint is created, but it's good
        # practice in case we later collect all ScanPoints first before computing. copy.copy is a "shallow copy"
        # meaning it gives a new object whos attribute slots point at whatever the attributes of the dvr instance
        # currently are. This is ok because, when changing the rotation angle, we create completely new objects
        # (`H`, `H0`, `V`, and `rotation_angle`) rather than mutating them in place. If we mutated them in place,
        # we'd have a problem since the copy will point to the mutated object. A deepcopy fixes this by creating
        # an object for each attribute, separate from its mutated future. In our case, this isn't needed.
        pt = ScanPoint(
            resonance_dvr=copy.copy(resonance_dvr), resonance_energy=renergy[0], resonance_state=rstate[:, 0],
            phi=phi, L=resonance_dvr.L,
            bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None,
            bound_energy=benergy, bound_state=bstate,
        )
        for name, fn in observables.items():
            results[name].append(fn(pt))
    return {name: np.array(val) for name, val in results.items()}
# ===============================================================================================================
#                                  PRINTING AND PLOTTING HELPERS
# ===============================================================================================================
def _format_training_string(name: str, data: np.ndarray):
    return (
        f"Training {name}: "
        f"({np.median(np.real(data))} +- {np.std(np.real(data))}) "
        f"+ i({np.median(np.imag(data))} +- {np.std(np.imag(data))})"
    )
    
def get_reference(data: np.ndarray, tol=1e-4):
    """
    A helper that returns the median of the data set if the std is below a tolerance.
    This function is meant to be used to get the "real resonance energy" and real values
    of the observables by averaging over the training data set.
    """
    if (np.std(np.real(data)) < tol) and (np.std(np.imag(data)) < tol):
        return np.median(data)
    else: 
        raise RuntimeError(f"Standard deviation of data not smaller than tolerance {tol:e}. "
                           f"Got {np.std(np.real(data))} +- i{np.std(np.imag(data))}"
                          )
    
def print_training_data(
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis: np.ndarray,
    observables: dict[str, Callable[[ScanPoint], complex]],
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
) -> dict[str, np.ndarray]:

    results = scan_vs_phi(
        resonance_dvr,
        resonance_energy,
        phis,
        observables,
        bound_dvr,
        bound_energy
    )
    for name, result in results.items():
        print(_format_training_string(name, result))
    return results

def print_quick_predict_resids(
    resonance_ec: ECSystem,
    phi_predict: complex,
    observables: dict[str, Callable[[ScanPoint], complex]],
    references: dict[str, complex],
    resonance_dvr: DVR,
    resonance_energy: complex,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
    relative: bool=True,
) -> tuple[dict[str, complex], dict[str, complex]]:
    """
    Note: this is inefficient because it computes the results twice (once for standard results
    and once for residuals), but given what I'm using this for, it's fine.
    """
    results = quick_predict(
        resonance_ec,
        phi_predict,
        observables,
        resonance_dvr,
        resonance_energy,
        bound_dvr,
        bound_energy
    )
    resid_results = quick_predict_resids(
        resonance_ec,
        phi_predict,
        observables,
        references,
        resonance_dvr,
        resonance_energy,
        bound_dvr,
        bound_energy,
        relative
    )

    for name, result in results.items():
        print(f"{name} at phi={phi_predict.real:.4f} +i{phi_predict.imag:.4f}: {result.real:.4f} + i{result.imag:.4f}")
        print(f"\t Residual: {resid_results[name].real:.4f} + i{resid_results[name].imag:.4f}")
    return results, resid_results
        
def plot_stats(fig, axs, xs, med, err68, err95, cutoff: int=0, color='red'):
    """ 
    Plots EC ensemble stats with error bands. Designed to plot data and error bars as number of training
    points varies, but `xs`, `med`, `err68`, and `err95` can be anything.

    Parameters
    ----------
    fig: plt.fig 
        the matplotlib figure.
    axs: tuple[plt.ax, plt.ax]
        Pair of matplotlib axes, one for the real component; one for the imaginary component.
    xs: np.ndarray
        The x-axis data. Shape (len(xs),).
    med: np.ndarray
        The median of whatever data set as x varies. Shape (len(xs),).
    err68: np.ndarray
        The 68% percentile error bands corresponding to the data in `med`. Shape (len(xs), 2). err68[:, 0]
        is the minimum for the error band as x varies, and err68[:, 1] is the maximum for the error band as 
        x varies.
    err95: np.ndarray
        Exactly as `err68` except for the 95% percentile error bands.
    cutoff: int
        Cuts off the first "cutoff" data points. For example, if cutoff=1, then only xs[1:], med[1:], etc. will be plotted. Default is no cutoff.
    color: str
        Color of the data. Default is red.
    """
    ax1, ax2 = axs
    xs, med, err68, err95 = xs[cutoff:], med[cutoff:], err68[cutoff:, :], err95[cutoff:, :]
    
    # real component 
    ax1.scatter(
        xs, np.real(med),
        color=color, 
        marker='x', 
        label='Predict (median)'
    )
    ax1.errorbar(
        xs, np.real(med),
        yerr=np.real(err68.T),
        fmt='none',
        ecolor=color, 
        elinewidth=2.0, 
        alpha=1.0, 
        label='Predict (68.2% int)'
    )
    ax1.errorbar(
        xs, np.real(med),
        yerr=np.real(err95.T),
        fmt='none',
        ecolor=color,
        elinewidth=2.0, 
        alpha=0.4, 
        label='Predict (95.4% int)'
    )
    
    # imag component
    ax2.scatter(
        xs, np.imag(med),
        color=color,
        marker='x', 
        label='Predict (median)'
    )
    ax2.errorbar(
        xs, np.imag(med),
        yerr=np.imag(err68.T),
        fmt='none', 
        ecolor=color, 
        elinewidth=2.0, 
        alpha=1.0, 
        label='Predict (68.2% int)'
    )
    ax2.errorbar(
        xs, np.imag(med),
        yerr=np.imag(err95.T),
        fmt='none', 
        ecolor=color, 
        elinewidth=2.0, 
        alpha=0.4, 
        label='Predict (95.4% int)'
    )

def get_plots_path():
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "plots"


# =====================================================================================================
#                                                SWEEP UTILS
# =======================================================================================================
# -------------------------------------------------------------------------------------------------------
# One scanned point and the observables that can be read off it
# ------------------------------------------------------------------------------------------------------
@dataclass
class ScanPoint:
    """
    Everything an observable function might need from one scan point.
    `resonance_dvr`, `resonance_energy`, `resonance_state` are the resonance-
    channel closest_to_resonance result. `bound_system`, `bound_energy`, `bound_state`
    are the matching bound-channel results, populated only if the scan was given a 
    bound_dvr + bound_energy (needed by obs_telem; None otherwise).

    `phi` and `L` are always populated (whichever one isn't being scanned is still whatever
    value the DVR system was built with).
    """
    resonance_dvr: DVR
    resonance_energy: complex
    resonance_state: np.ndarray

    phi: complex
    L: float

    bound_dvr: DVR | None = None
    bound_energy: complex | None = None
    bound_state: np.ndarray | None = None

def obs_energy(pt: ScanPoint) -> complex:
    """ Observable: the resonance energy itself. """
    return pt.resonance_energy

def obs_rr(pt: ScanPoint, rotate_rr: bool=True) -> complex:
    """
    Observable: (r^2) of the resonance state. 
    Default usage is `{"rr": obs_rr}`. For the rare non-default case, use
    `functools.partial(obs_rr, rotate_rr=False)`.
    """
    return pt.resonance_dvr.compute_rr(pt.resonance_state, rotate_rr=rotate_rr)

def obs_telem(pt: ScanPoint, operator: str="E1", phase_fixed: bool=True) -> complex:
    """ 
    Observable: (psi_res | operator | psi_bound). Requires the scan call
    to be given bound_dvr and bound_energy. Default usage is `{"telem": obs_telem}`.
    For the rare non-default case, use `functools.partial(obs_telem, operator=...)`.

    Note: the transition matrix element is determined under the c-product up to a sign. 
    Thus, the sign of the transition matrix element below should be "phase-fixed" according
    to some reference value. This function defaults to fixing the sign such that the real component is positive.
    """
    if pt.bound_dvr is None:
        raise ValueError(
            "obs_telem needs a bound state -- pass bound_dvr and "
            "bound_energy to scan_vs_phi / scan_vs_L"
        )
    if (not np.isclose(pt.bound_dvr.L, pt.L)) or (pt.resonance_dvr.n != pt.bound_dvr.n):
        raise ValueError(
            f"Bound state DVR and resonance DVR need to have the same system length and same number of mesh points. "
            f"Got ({pt.bound_dvr.L:.5f}, {pt.bound_dvr.n}) vs ({pt.L:.5f}, {pt.resonance_dvr.n})."
        )
    telem = DVR.compute_transition_mat_element(
        pt.L,
        pt.bound_state,
        pt.resonance_state,
        rotation_angle=pt.phi,
        operator=operator
    )
    if phase_fixed:
        telem *= np.sign(np.real(telem))
    
    return telem

# ------------------------------------------------------------------------------------------
# Scan engines: one diagonalization per point, many observables per point
# ------------------------------------------------------------------------------------------

def scan_vs_phi(
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis: np.ndarray,
    observables: dict[str, Callable[[ScanPoint], complex]],
    bound_dvr: DVR | None = None,
    bound_energy: complex | None = None,
) -> dict[str, np.ndarray]:
    """
    Scans phi, evaluating every observable in `observables` at each scan point. Also includes a "bound_energy"
    key that tracks the bound state energy over every phi in `phis`.
    """    
    resonance_energy = complex(resonance_energy)
    phis = np.asarray(phis)
    
    if bound_dvr is not None:
        bound_energy = complex(bound_energy)
        if bound_dvr.L != resonance_dvr.L:
            raise ValueError(
                f"The bound system and resonance system need to have the same system length. "
                f"Got {bound_dvr.L} vs {resonance_dvr.L}."
            )
        if bound_dvr.n != resonance_dvr.n:
            raise ValueError(
                f"The bound system and resonance system need to have the same number of mesh points. "
                f"Got {bound_dvr.n} vs {resonance_dvr.n}."
            )
            
    results = {name: [] for name in observables}
    results["bound_energies"] = []
    for phi in phis:
        phi = complex(phi)
        resonance_dvr.reset_rotation_angle(phi)
        renergy, rstate = resonance_dvr.closest_to_resonance(resonance_energy)

        benergy = bstate = None
        if bound_dvr is not None:
            bound_dvr.reset_rotation_angle(phi)
            benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
            benergy, bstate = benergy[0], bstate[:, 0]
            results["bound_energies"].append(benergy)

        # copy.copy copies the dvr instance at it's current state (current rotation angle). This isn't strictly
        # necessary since we immediately compute observables right after the ScanPoint is created, but it's good
        # practice in case we later collect all ScanPoints first before computing. copy.copy is a "shallow copy"
        # meaning it gives a new object whos attribute slots point at whatever the attributes of the dvr instance
        # currently are. This is ok because, when changing the rotation angle, we create completely new objects
        # (`H`, `H0`, `V`, and `rotation_angle`) rather than mutating them in place. If we mutated them in place,
        # we'd have a problem since the copy will point to the mutated object. A deepcopy fixes this by creating
        # an object for each attribute, separate from its mutated future. In our case, this isn't needed.
        pt = ScanPoint(
            resonance_dvr=copy.copy(resonance_dvr), resonance_energy=renergy[0], resonance_state=rstate[:, 0],
            phi=phi, L=resonance_dvr.L,
            bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None,
            bound_energy=benergy, bound_state=bstate,
        )
        for name, fn in observables.items():
            results[name].append(fn(pt))
    return {name: np.array(val) for name, val in results.items()}

def scan_vs_L(
    rotation_angle: complex,
    resonance_energy: complex,
    Ls: np.ndarray,
    mesh_width: float,
    observables: dict[str, Callable[[ScanPoint], complex]],
    resonance_potential: Callable[[complex], complex] | None=None,
    resonance_l: int=0,
    bound_energy: complex | None=None,
    bound_potential: Callable[[complex], complex] | None=None,
    bound_l: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Scans L while holding the DVR mesh spacing dr = L/(n+1) fixed (n is recomputed at each L).
    One diagonalization per L, every observable evaluated against it. Also includes a "bound_energies"
    key that tracks the bound state energy over every L in `Ls`.
    """
    rotation_angle = complex(rotation_angle)
    resonance_energy = complex(resonance_energy)
    Ls = np.asarray(Ls)
    mesh_width = float(mesh_width)
    resonance_l = int(resonance_l)

    results = {name: [] for name in observables}
    bound_energies = []
    ns, actual_mesh_widths = [], []
    for L in Ls:
        L = float(L)
        n = max(int(round(L / mesh_width)) - 1, 1)

        ns.append(n)
        actual_mesh_widths.append(float(L) / (n + 1))
        
        resonance_dvr = DVR(n, L, rotation_angle, resonance_potential, resonance_l)
        renergy, rstate = resonance_dvr.closest_to_resonance(resonance_energy)

        bound_dvr = benergy = bstate = None
        if bound_energy is not None:
            bound_energy = complex(bound_energy)
            bound_l = int(bound_l)
            bound_dvr = DVR(n, L, rotation_angle, bound_potential, bound_l)
            benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
            benergy, bstate = benergy[0], bstate[:, 0]
            bound_energies.append(benergy)

        pt = ScanPoint(
            resonance_dvr=resonance_dvr, resonance_energy=renergy[0], resonance_state=rstate[:, 0],
            phi=rotation_angle, L=L,
            bound_dvr=bound_dvr, bound_energy=benergy, bound_state=bstate
        )
        for name, fn in observables.items():
            results[name].append(fn(pt))
            
    results = {name: np.array(val) for name, val in results.items()}
    results["ns"] = np.array(ns, dtype=int)
    results["actual_mesh_widths"] = np.array(actual_mesh_widths)
    results["bound_energies"] = np.array(bound_energies)
    return results
    
# -----------------------------------------------------------------------------------------------
# Plateau / good-range detection

# NOTE: many, not all, of these functions 
# ----------------------------------------------------------------------------------------------

def get_phi_bounds(resonance_energy: complex, phi_max: float=np.pi/4) -> tuple[float, float]:
    """
    Theoretical (phi_min, phi_max) window for a resonance under uniform complex scaling.
    phi_min = |arg(E_res)| / 2 (rotation needed to expose the pole);
    phi_max defaults to pi/4, correct for any Gaussian-type potential family
    (V(r) ~ exp(-r^2 e^{2i theta}) needs theta < pi/4 to stay integrable).
    For other potential families, phi_max needs to be supplied explicitly.

    We allow the rotation angle to run complex, but this returns the Re(phi) bounds, so
    they remain floats.
    """
    resonance_energy = complex(resonance_energy)
    return abs(np.angle(resonance_energy)) / 2, float(phi_max)

def _relative_diffs(ys: np.ndarray, floor: float=1e-12) -> np.ndarray:
    ys = np.asarray(ys)
    denom = np.maximum(np.abs(np.real(ys)[:-1]), floor) + 1j*np.maximum(np.abs(np.imag(ys)[:-1]), floor)
    diffs = np.diff(ys)
    return np.abs(np.real(diffs)) / np.real(denom) + 1j*np.abs(np.imag(diffs)) / np.imag(denom)

def _longest_run(good: np.ndarray) -> tuple[int, int] | None:
    """
    Longest contiguous stretch of True in a 1D boolean array.
    Returns (start_index, length) or None if `good` is all False.
    """
    best_start, best_len, current_start, current_len = None, 0, None, 0
    for i, g in enumerate(good):
        if g:
            if current_start is None:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
        else:
            current_start = None
            current_len = 0

    return None if best_start is None else (best_start, best_len)

def longest_flat_run(
    xs: np.ndarray,
    ys: np.ndarray,
    rel_tol: complex=1e-4+1j*1e-4,
    min_run: int=5,
) -> tuple[float, float] | None:
    """
    Finds the longest contiguous stretch of xs over which successive relative
    changes in ys stays below rel_tol.

    NOTE: while this function works when `xs` are complex, it's generally not the way
    you'd want to find the longest flat run. This function won't search through the real
    and imaginary component independently. Instead, it collapses the values into one array
    and searches along that axis.
    """
    rel_tol = complex(rel_tol)
    min_run = int(min_run)
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must be the same length, got {len(xs)} vs {len(ys)}.")

    rel_diffs = _relative_diffs(ys)
    run = _longest_run( (np.real(rel_diffs) < rel_tol.real) & (np.imag(rel_diffs) < rel_tol.imag))
    if run is None or run[1] < min_run:
        return None

    start, length = run
    return float(xs[start]), float(xs[start + length])

def joint_flat_run(
    xs: np.ndarray,
    series: dict[str, np.ndarray],
    rel_tol: complex | dict[str, complex]=1e-4+1j*1e-4,
    min_run: int=5,
) -> tuple[float, float] | None:
    """
    Finds the longest contiguous stretch of xs over which success relative
    changes in each array in ys stays below rel_tol. `rel_tol` can be a single
    float (the same tolerance for each array in `series`) or a dictionary with
    keys matching those in series (applying a different tolerance for each array
    in `series`).

    NOTE: see `longest_flat_run`'s note.
    """
    xs = np.asarray(xs)
    tols = rel_tol if isinstance(rel_tol, dict) else {name: complex(rel_tol) for name in series.keys()}
    
    diff_bool_list = []
    for name, ys in series.items():
        ys = np.asarray(ys)
        if len(ys) != len(xs):
            raise ValueError(f"xs and ys must be the same length, got {len(xs)} vs {len(ys)}.")

        # compute the run of bools associated with if _relative_diffs(ys) < rel_tol
        rtol = complex(tols[name])
        rel_diffs = _relative_diffs(ys)
        diff_bool = (np.real(rel_diffs) < rtol.real) & (np.imag(rel_diffs) < rtol.imag)
        diff_bool_list.append(np.array(diff_bool))

    # compute logical and of all run arrays together (each element is True only if 
    # the relative difference of each set of ys in `series` is below its respective
    # relative at that x value 
    final_diff_bool_list = np.logical_and.reduce(diff_bool_list)
    final_run = _longest_run(final_diff_bool_list)
    if final_run is None or final_run[1] < min_run:
        return None
    start, length = final_run 
    return float(xs[start]), float(xs[start + length])

# -------------------------------------------------------------------------------------------------------------
# (L, mesh_width) and phi_range search

# NOTE: these functions assume that `phis` are pure real (sort of. some of them technically work when phis are complex,
# but they're generally not what you want.
# -------------------------------------------------------------------------------------------------------------
def _badness(series: dict[str, np.ndarray]) -> float:
    """
    A single real score for how flat a set of observable
    series are. Lower is better.

    ns / actual_mesh_widths (scan_vs_L's extra keys) are
    skipped since they aren't observables to score. bound_energies
    is also skipped since it's nearly completely real, so the relative
    residual will explode badness artificially.
    """
    total = 0.0
    for name, ys in series.items():
        if name in ("ns", "actual_mesh_widths", "bound_energies"):
            continue
        d = _relative_diffs(np.asarray(ys))
        total += float(np.median(np.abs(d.real)) + np.median(np.abs(d.imag)))
    return total

def search_mesh_width(
    resonance_potential: Callable[[complex], complex],
    resonance_energy: complex,
    resonance_l: int,
    L: float,
    mesh_widths: np.ndarray,
    phi_lo: float,
    phi_hi: float,
    phi_scan: int=30,
    bound_potential: Callable[[complex], complex] | None=None,
    bound_energy: complex | None=None,
    bound_l: int | None=None,
) -> list[dict]:
    """
    Stage 1 of the (L, mesh_width) search: at fixed L, scores each candidate
    mesh width in `mesh_widths` by how flat E and rr (and telem if a bound
    state is supplied) are over phi in [phi_lo, phi_hi].

    Returns a list of dicts (one per candidate) sorted best (lowest badness)
    first: {"mesh_widths", "n", "badness", "series"}.

    NOTE: This method assumes the rotation angle is completely real. If one wants to 
    extend this to complex values of the rotation angle, one needs to pass in the real
    and imaginary parts separately.
    """
    resonance_energy = complex(resonance_energy)
    resonance_l = int(resonance_l)
    L = float(L)
    mesh_widths = np.asarray(mesh_widths)
    phi_lo, phi_hi = float(phi_lo), float(phi_hi)
    phi_scan = int(phi_scan)
    
    phis = np.linspace(phi_lo, phi_hi, phi_scan)
    observables = {"E": obs_energy, "rr": obs_rr}
    if bound_energy is not None:
        bound_energy = complex(bound_energy)
        bound_l = int(bound_l)
        observables["telem"] = obs_telem

    results = []
    for mesh_width in mesh_widths:
        mesh_width = float(mesh_width)
        n = max(int(round(L / mesh_width)) - 1, 1)

        resonance_dvr = DVR(n, L, phi_lo, resonance_potential, resonance_l)
        bound_dvr = None
        if bound_energy is not None:
            bound_dvr = DVR(n, L, phi_lo, bound_potential, bound_l)

        series = scan_vs_phi(
            resonance_dvr, resonance_energy, phis, observables,
            bound_dvr=bound_dvr, bound_energy=bound_energy,
        )
        results.append({
            "mesh_width": mesh_width, "n": n,
            "badness": _badness(series), "series": series,
        })
    return sorted(results, key=lambda r: r["badness"])
    
def search_L(
    resonance_potential: Callable[[complex], complex],
    resonance_energy: complex,
    resonance_l: int,
    Ls: np.ndarray,
    mesh_width: float,
    phi_lo: float,
    phi_hi: float,
    phi_scan: int=30,
    bound_potential: Callable[[complex], complex] | None=None,
    bound_energy: complex | None=None,
    bound_l: int | None=None,
) -> list[dict]:
    """
    Stage 2 of the (L, mesh_width) search: at a fixed mesh_width (the 
    winner from search_mesh_widths, normally), scores each candidate L
    in `Ls`.

    Returns a list of dict (one per candidate), sorted best first:
    {"L", "n", "badness", "series"}.

    NOTE: This method assumes the rotation angle is completely real. If one wants to 
    extend this to complex values of the rotation angle, one needs to pass in the real
    and imaginary parts separately.
    """
    resonance_energy = complex(resonance_energy)
    resonance_l = int(resonance_l)
    Ls = np.asarray(Ls)
    mesh_width = float(mesh_width)
    phi_lo, phi_hi = float(phi_lo), float(phi_hi)
    phi_scan = int(phi_scan)
    
    observables = {"E": obs_energy, "rr": obs_rr}
    if bound_energy is not None:
        bound_energy = complex(bound_energy)
        bound_l = int(bound_l)
        observables["telem"] = obs_telem

    results = []
    for L in Ls:
        L = float(L)
        n = max(int(round(L / mesh_width)) - 1, 1)
        phis = np.linspace(phi_lo, phi_hi, phi_scan)

        resonance_dvr = DVR(n, L, phi_lo, resonance_potential, resonance_l)
        bound_dvr = None
        if bound_energy is not None:
            bound_dvr = DVR(n, L, phi_lo, bound_potential, bound_l)

        series = scan_vs_phi(
            resonance_dvr, resonance_energy, phis, observables,
            bound_dvr=bound_dvr, bound_energy=bound_energy,
        )
        results.append({
            "L": L, "n": n,
            "badness": _badness(series), "series": series,
        })

    return sorted(results, key=lambda r: r["badness"])

def search_optimal_phirange(
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis: np.ndarray,
    min_run: int=20,
    rel_tol: complex=1e-3+1j*1e-3,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
) -> tuple[float, float]:
    """ 
    phi_range search. At a fixed L and mesh_width (the winners from
    `search_mesh_width` and `search_L`, normally), finds the optimal
    phi values between `phi_lo` and `phi_hi` using `joint_flat_run`.

    Returns tuple[float, float]: [optimal_phi_min, optimal_phi_max].
    """
    resonance_energy = complex(resonance_energy)
    phis = np.asarray([complex(phi) for phi in phis])
    min_run = int(min_run)
    rel_tol = complex(rel_tol)

    observables = {"E": obs_energy, "rr": obs_rr}
    if bound_energy is not None:
        bound_energy = complex(bound_energy)
        observables["telem"] = obs_telem

    results = scan_vs_phi(
        resonance_dvr, resonance_energy, phis, observables,
        bound_dvr=bound_dvr, bound_energy=bound_energy
    )
    return joint_flat_run(
        phis,
        series=results,
        rel_tol=rel_tol,
        min_run=min_run
    )

# ==============================================================================================================
#                                   CONVENIENCE METHODS FOR PREDICTION AND TRAINING DATA
# ==============================================================================================================
# ---------------------------------------------------------------------------------------------------------------
# One prediction point and the observables that can be read off it
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PredictPoint:
    """
    Everything an observable function might need from one prediction point.
    """
    predicted_resonance_energy: complex
    predicted_resonance_state: np.ndarray

    phi_predict: float
    resonance_ec: ECSystem

    bound_dvr: DVR | None = None
    bound_energy: complex | None = None
    bound_state: np.ndarray | None = None

def predict_energy(pt: PredictPoint) -> complex:
    """ Predicted observable: predicted resonance energy. """
    return pt.predicted_resonance_energy

def predict_rr(pt: PredictPoint, rotate_rr: bool=True) -> complex:
    """ Predicted observable: predicted (r^2). """
    resonance_dvr = pt.resonance_ec.system
    resonance_dvr.reset_rotation_angle(pt.phi_predict)
    return resonance_dvr.compute_rr(pt.predicted_resonance_state, rotate_rr=rotate_rr)

def predict_telem(pt: PredictPoint, operator: str="E1", phase_fixed: bool=True) -> complex:
    """ 
    Predicted observable: predicted (psi_res | operator | psi_bound). As in `obs_telem`, this
    function defaults to phase-fixing the element such that the real component is positive.
    """
    resonance_dvr = pt.resonance_ec.system
    L = resonance_dvr.L
    if pt.bound_dvr is None:
        raise ValueError(
            "predict_telem needs a bound state -- pass bound_dvr and "
            "bound_energy to quick_predict_resids."
        )
    if (not np.isclose(pt.bound_dvr.L, L)) or (resonance_dvr.n != pt.bound_dvr.n):
        raise ValueError(
            f"Bound state DVR and resonance DVR need to have the same system length and same number of mesh points. "
            f"Got ({pt.bound_dvr.L:.5f}, {pt.bound_dvr.n}) vs ({L:.5f}, {resonance_dvr.n})."
        )
    telem = DVR.compute_transition_mat_element(
        L,
        pt.bound_state,
        pt.predicted_resonance_state,
        rotation_angle=pt.phi_predict,
        operator=operator,
    )
    if phase_fixed:
        telem *= np.sign(np.real(telem))
    return telem

# ------------------------------------------------------------------------------------------------------------------------
# Convenience prediction engines along with statistics
# ------------------------------------------------------------------------------------------------------------------------

def _residual(reference: complex, prediction: complex) -> complex:
    """ Compute the component-wise residual between `prediction` and `reference`. """
    diff = prediction - reference
    return np.abs(diff.real) + 1j*np.abs(diff.imag)
    
def _relative_residual(reference: complex, prediction: complex, tol: float=1e-9) -> complex:
    """ Compute the component-wise relative residual between `prediction` and `reference`. """
    diff = prediction - reference
    if (np.abs(reference.real) < tol) or (np.abs(reference.imag) < tol):
        raise ValueError(
            f"Reference can't be zero in the real or imaginary component for relative residual. "
            f"Got a real or imaginary component of 0 within tolerance {tol:e}. Use `_residual` "
            f"instead."
        )
    return np.abs(diff.real) / reference.real + 1j*np.abs(diff.imag) / reference.imag

def quick_predict(
    resonance_ec: ECSystem,
    phi_predict: float,
    observables: dict[str, Callable[[ScanPoint], complex]],
    resonance_dvr: DVR,
    resonance_energy: complex,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
) -> dict[str, complex]:
    """
    Quickly predict many observables using EC. Returns dictionary of results
    where the keys share the same names as those in `observables` and the val is the computed observable.
    Also includes a key "bound_energy" which includes the bound energy at `phi_predict`.
    """
    phi_predict = float(phi_predict)
    resonance_energy = complex(resonance_energy)

    predicted_resonance_energy, predicted_resonance_state = resonance_ec.predict_DVR_state_at(phi_predict)
    
    results = {} 
    benergy = bstate = None
    if bound_dvr is not None:
        # get bound state (no need to use EC here). This shouldn't change at all.
        bound_dvr.reset_rotation_angle(phi_predict)
        bound_energy = complex(bound_energy)
        benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
        benergy, bstate = benergy[0], bstate[:, 0]
        results["bound_energy"] = benergy

    pt = PredictPoint(
        predicted_resonance_energy, predicted_resonance_state, phi_predict, copy.copy(resonance_ec),
        bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None, 
        bound_energy=benergy, bound_state=bstate
    )
    for name, fn in observables.items():
        results[name] = fn(pt)
    return results

def quick_predict_resids(
    resonance_ec: ECSystem,
    phi_predict: float,
    observables: dict[str, Callable[[ScanPoint], complex]],
    references: dict[str, complex | None],
    resonance_dvr: DVR,
    resonance_energy: complex,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
    relative: bool=True,
) -> dict[str, complex]:
    """
    Quickly predict many observables using EC and compute residuals. Returns dictionary of results
    where the keys share the same names as those in `observables` and the item is the residual of the
    result. Residuals are computed against the items in `references`. `references` and `observables`
    should share the same keys. Also includes a key "bound_energy" which includes the residual bound energy
    at `phi_predict`.
    """
    phi_predict = float(phi_predict)
    resonance_energy = complex(resonance_energy)

    predicted_resonance_energy, predicted_resonance_state = resonance_ec.predict_DVR_state_at(phi_predict)

    results = {}
    benergy = bstate = None
    if bound_dvr is not None:
        # get bound state (no need to use EC here). This shouldn't change at all.
        bound_dvr.reset_rotation_angle(phi_predict)
        bound_energy = complex(bound_energy)
        benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
        benergy, bstate = benergy[0], bstate[:, 0]
        if relative:
            # we add 1j to avoid dividing by zero
            results["bound_energy"] = _relative_residual(bound_energy + 1j, benergy).real
        else:
            results["bound_energy"] = _residual(bound_energy, benergy).real

    pt = PredictPoint(
        predicted_resonance_energy, predicted_resonance_state, phi_predict, copy.copy(resonance_ec),
        bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None, 
        bound_energy=benergy, bound_state=bstate
    )

    for name, fn in observables.items():
        result = fn(pt)
        ref = references[name]
        if relative:
            results[name] = _relative_residual(ref, result)
        else:
            results[name] = _residual(ref, result)
    return results

# =============================================================================================================
#                                                 EC ENSEMBLE HELPERS 
# =============================================================================================================
def decompose_stats(data: list):
    """
    A helper for unraveling the data output by EC Ensemble methods like `compute_resid_stats` and `compute_stats`
    when used in a loop (e.g. when computing residual stats as we vary the number of training points.

    Parameters
    ----------
    data: np.ndarray
        The data to unravel. Assumes `data` is structured as follows: data[i] = tuple[median, err68, err95] for EC Ensemble
        system i.

    Returns
    -------
    med: np.ndarray
        Array of medians over all i in data[i]. Shape (len(data),).
    err68: np.ndarray
        Array of 68% percentile error bands over all i in data[i]. Shape (len(data), 2).
    err95: np.ndarray
        Exactly as err68 but for the 95% percentile error bands.
    """
    
    med = [r[0] for r in data]
    err68 = [r[1] for r in data]
    err95 = [r[2] for r in data]
    return np.array(med), np.array(err68), np.array(err95)