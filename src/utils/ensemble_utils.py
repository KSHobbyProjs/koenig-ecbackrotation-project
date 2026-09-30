"""
ensemble_utils.py

A set of convenience methods for predicting observables and error bands using an ECEnsemble.
"""

import numpy as np

from ..dvr import DVR
from ..ecensemble import ECEnsemble, StatPoint
from .ec_utils import residual, relative_residual

from typing import Callable
import copy

from dataclasses import dataclass

# ========================================================================================
#                           ENSEMBLE POINT AND OBSERVABLES
# =========================================================================================
@dataclass
class EnsemblePoint:
    """
    Everything an observable function might need to compute observables over a batch of data.
    `predicted_resonance_states` should be an array of shape (dvr dim, n_ecsystems).
    """
    predicted_resonance_energies: np.ndarray
    predicted_resonance_states: np.ndarray

    phi_predict: complex
    resonance_dvr: DVR

    bound_dvr: DVR | None = None
    bound_energy: complex | None = None
    bound_state: np.ndarray | None = None

def ensemble_energies(ep: EnsemblePoint) -> np.ndarray:
    return ep.predicted_resonance_energies

def ensemble_rrs(ep: EnsemblePoint, rotate_rr: bool=True) -> np.ndarray:
    ep.resonance_dvr.reset_rotation_angle(ep.phi_predict)
    return np.array([
        ep.resonance_dvr.compute_rr(s, rotate_rr=rotate_rr) for s in ep.predicted_resonance_states.T
    ])    

def ensemble_telems(ep: EnsemblePoint, operator: str="E1", phase_fixed: bool=True) -> np.ndarray:
    L = ep.resonance_dvr.L
    if ep.bound_dvr is None:
        raise ValueError(
            "predict_telem needs a bound state -- pass bound_dvr and "
            "bound_energy to quick_predict_resids."
        )
    if (not np.isclose(ep.bound_dvr.L, L)) or (ep.resonance_dvr.n != ep.bound_dvr.n):
        raise ValueError(
            f"Bound state DVR and resonance DVR need to have the same system length and same number of mesh points. "
            f"Got ({ep.bound_dvr.L:.5f}, {ep.bound_dvr.n}) vs ({L:.5f}, {ep.resonance_dvr.n})."
        )
    telems = []
    for s in ep.predicted_resonance_states.T:
        telem = DVR.compute_transition_mat_element(
            L,
            ep.bound_state,
            s,
            rotation_angle=ep.phi_predict,
            operator=operator
        )
        if phase_fixed:
            telem *= np.sign(np.real(telem))
        telems.append(telem)
    return np.array(telems)

def ensemble_density(ep: EnsemblePoint, x_plot: np.ndarray | None=None) -> np.ndarray:
    ep.resonance_dvr.reset_rotation_angle(ep.phi_predict)
    return np.array([
        ep.resonance_dvr.compute_density(s, x_plot) for s in ep.predicted_resonance_states.T
    ]).T
    
# -------------------------------------------------------------------------------------------
# One ensemble prediction point and observables that can be read off it
# ------------------------------------------------------------------------------------------
def quick_predict_ensemble(
    ecensemble: ECEnsemble,
    phi_predict: complex,
    observables: dict[str, Callable[[EnsemblePoint], complex]],
    resonance_dvr: DVR, 
    resonance_energy: complex,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
) -> dict[str, StatPoint]:
    
    phi_predict=complex(phi_predict)
    resonance_energy = complex(resonance_energy)
    
    energies, dvr_states = ecensemble.predict_DVR_states_at(phi_predict)

    results = {}
    benergy = bstate = None
    if bound_dvr is not None:
        bound_dvr.reset_rotation_angle(phi_predict)
        benergy, bstate = bound_dvr.closest_to_resonance(bound_energy)
        benergy, bstate = benergy[0], bstate[:, 0]

    ep = EnsemblePoint(
        predicted_resonance_energies=energies, predicted_resonance_states=np.copy(dvr_states), 
        phi_predict=phi_predict, resonance_dvr=copy.copy(resonance_dvr),
        bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None,
        bound_energy=benergy, bound_state=bstate
    )
    for name, fn in observables.items():
        results[name] = fn(ep)
    return results

# =======================================================================================================
#                                            HELPERS
# =======================================================================================================
def quick_stats(
    prediction: dict[str, np.ndarray],
    reference: dict[str, complex] | None=None,
    *,
    relative: bool=False,
    axis: dict[str, int] | int=0
) -> dict[str, StatPoint]:
    """
    Computes the med, err68, err95 of every observable in prediction. If a reference is given, the residual
    stats will be computed. If relative=True, the relative residual stats will be computed. axis determines 
    the axis along which the stats will be computed for each observable. Default is zero for all.
    """
    axes = axis if isinstance(axis, dict) else {name: int(axis) for name in prediction.keys()} 
    
    stats = {}
    if reference is None:
        for name, val in prediction.items():
            stats[name] = ECEnsemble.compute_stats(val, axes[name])
        return stats

    for name, vals in prediction.items():
        r = reference[name]
        if relative:
            resids = relative_residual(r, vals)
        else:
            resids = residual(r, vals)
        stats[name] = ECEnsemble.compute_stats(resids, axes[name])
    return stats
            
    