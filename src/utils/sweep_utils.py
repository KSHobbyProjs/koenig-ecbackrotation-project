"""
sweep_utils.py

A series of utility methods for computing observables over
a range of phi and L values.
"""

import numpy as np
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

def linesweep_vs_phi(
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

def gridsweep_vs_phi():
    pass