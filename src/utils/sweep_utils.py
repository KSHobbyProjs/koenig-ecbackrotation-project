"""
sweep_utils.py

A series of utility methods for computing observables over
a range of phi and L values.
"""

import numpy as np
from ..dvr import DVR
from typing import Callable
from dataclasses import dataclass
import copy

# =====================================================================================================
#                                        SCANNING POINT and OBSERVABLES
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

# ==========================================================================================
#                                        SCAN ENGINES 
# ==========================================================================================

def linesweep_phi(
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis: np.ndarray,
    observables: dict[str, Callable[[ScanPoint], complex]],
    bound_dvr: DVR | None = None,
    bound_energy: complex | None = None,
) -> dict[str, np.ndarray]:
    """
    Scans phi, evaluating every observable in `observables` at each scan point. Also includes a "bound_energy"
    key that tracks the bound state energy over every phi in `phis`. `phis` can be complex or pure real, but note
    that this assumes phis is a 1D array.
    
    Returns
    -------
    dict[str, np.ndarray]:
        Dictionary of results. Keys match those of `observables`. Also includes a key "bound_energies"
        that tracks the bound energies (if bound_dvr is supplied) over the sweep.
    """    
    resonance_energy = complex(resonance_energy)
    phis = np.asarray(phis)
    if phis.ndim != 1:
        raise ValueError(f"linesweep_phi assumes phis is a 1D array. Got array of shape {phis.shape}.")

    results = {name: [] for name in observables}
    if bound_dvr is not None:
        results["bound_energies"] = []
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

def gridsweep_phi(    
    resonance_dvr: DVR,
    resonance_energy: complex,
    phis_real: np.ndarray,
    phis_imag: np.ndarray,
    observables: dict[str, Callable[[ScanPoint], complex]],
    bound_dvr: DVR | None = None,
    bound_energy: complex | None = None,
) -> dict[str, np.ndarray]:
    """
    Scans phi, evaluating every observable in `observables` at each scan point. Also includes a "bound_energy"
    key that tracks the bound state energy over every phi. `phis_real` and `phis_imag` should both be completely real 1D arrays. The function will scan over all complex phi combinations of phis_real and phis_imag. I.e., scans over each phi in phis[i,j] = phis_real[i] + 1j*phis_imag[j]. The vals of the returned dictionary have shape (len(phis_real), len(phis_imag)) such that 
    dict["name"][i, j] corresponds to the "name" observable at phi = phi_real[i] + 1j*phi_imag[j]. Also includes a key "bound_energies"
        that tracks the bound energies (if bound_dvr is supplied) over the sweep.
    
    Returns
    -------
    dict[str, np.ndarray]:
        Dictionary of results. Keys match those of `observables`. Array has shape (
    """    
    phis_real = np.asarray(phis_real)
    phi_imag = np.asarray(phis_imag)
    
    if phis_real.ndim != 1 or phis_imag.ndim != 1:
        raise ValueError(f"gridsweep_phi expects phis_real and phis_imag to be 1D arrays. Got shapes {phis_real.shape} and {phis_imag.shape}.")
    if (not np.issubdtype(phis_real.dtype, np.floating)) or (not np.issubdtype(phis_imag.dtype, np.floating)):
        raise ValueError(f"gridsweep_phi expects phis_real and phis_imag to both be 1D arrays of pure floats. Got types {phis_real.dtype} and {phis_imag.dtype}.")

    results = {name: np.zeros((len(phis_real), len(phis_imag)), dtype=np.complex128) for name in observables}
    resonance_energy = complex(resonance_energy)
    if bound_dvr is not None:
        bound_energy = complex(bound_energy)
        results["bound_energies"] = np.zeros((len(phis_real), len(phis_imag)), dtype=np.complex128)
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
    
    xv, yv = np.meshgrid(phis_real, phis_imag, indexing='ij')
    for i in range(len(phis_real)):
        for j in range(len(phis_imag)):
            phir = xv[i, j]
            phii = yv[i, j]
            phi = phir + 1j*phii
            
            resonance_dvr.reset_rotation_angle(phi)
            renergy, rstate = resonance_dvr.closest_to_resonance(resonance_energy)

            benergy = bstate = None
            if bound_dvr is not None:
                bound_dvr.reset_rotation_angle(phi)
                benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
                benergy, bstate = benergy[0], bstate[:, 0]
                results["bound_energies"][i, j] = benergy

            pt = ScanPoint(
                resonance_dvr=copy.copy(resonance_dvr), resonance_energy=renergy[0], resonance_state=rstate[:, 0],
                phi=phi, L=resonance_dvr.L,
                bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None,
                bound_energy=benergy, bound_state=bstate,
            )
            for name, fn in observables.items():
                results[name][i,j] = fn(pt)
    return results

def linesweep_L(
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

# ======================================================================================================
#                                       PLATEAU / GOOD-RANGE DETECTION
# ======================================================================================================
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


# =============================================================================================================
#                                          (L, MESH WIDTH, PHI RANGE) SEARCH
# =============================================================================================================
# -------------------------------------------------------------------------------------------------------------
# NOTE: these functions should only be used with arrays of real rotation angles. If you're using complex rotation
# angles, the real and imaginary components should be swept through separately. 
    
# These functions sweep over the chosen parameter (mesh width, L, etc.), for each value of the parameter, the function
# scans over phi, measuring how much a set of observables deviate from a constant value as phi changes. This deviation is
# measured as the median of the relative difference of the observable over phi (we use median so that large differences near
# the edge of the phi range don't contribute. We call this measure "badness". 
# The parameter value that has the smallest "badness" is deemed the best.
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

def search_optimal_phireal_range(
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