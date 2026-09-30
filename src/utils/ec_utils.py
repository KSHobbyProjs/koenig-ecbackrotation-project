"""
predict_utils.py

A set of convenience methods for getting predictions.
"""

import numpy as np

from ..dvr import DVR
from ..ec import ECSystem
from typing import Callable
import copy

from dataclasses import dataclass


# ===============================================================================================================
#                                           PREDICTION POINT and OBSERVABLES
# ===============================================================================================================
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

    phi_predict: complex
    resonance_dvr: DVR

    bound_dvr: DVR | None = None
    bound_energy: complex | None = None
    bound_state: np.ndarray | None = None

def predict_energy(pt: PredictPoint) -> complex:
    """ Predicted observable: predicted resonance energy. """
    return pt.predicted_resonance_energy

def predict_rr(pt: PredictPoint, rotate_rr: bool=True) -> complex:
    """ Predicted observable: predicted (r^2). """
    pt.resonance_dvr.reset_rotation_angle(pt.phi_predict)
    return pt.resonance_dvr.compute_rr(pt.predicted_resonance_state, rotate_rr=rotate_rr)

def predict_telem(pt: PredictPoint, operator: str="E1", phase_fixed: bool=True) -> complex:
    """ 
    Predicted observable: predicted (psi_res | operator | psi_bound). As in `obs_telem`, this
    function defaults to phase-fixing the element such that the real component is positive.
    """
    pt.resonance_dvr.reset_rotation_angle(pt.phi_predict)
    L = pt.resonance_dvr.L
    if pt.bound_dvr is None:
        raise ValueError(
            "predict_telem needs a bound state -- pass bound_dvr and "
            "bound_energy to quick_predict_resids."
        )
    if (not np.isclose(pt.bound_dvr.L, L)) or (pt.resonance_dvr.n != pt.bound_dvr.n):
        raise ValueError(
            f"Bound state DVR and resonance DVR need to have the same system length and same number of mesh points. "
            f"Got ({pt.bound_dvr.L:.5f}, {pt.bound_dvr.n}) vs ({L:.5f}, {pt.resonance_dvr.n})."
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

def predict_density(pt: PredictPoint, x_plot: np.ndarray | None=None) -> np.ndarray:
    pt.resonance_dvr.reset_rotation_angle(pt.phi_predict)
    return pt.resonance_dvr.compute_density(pt.predicted_resonance_state, x_plot)

# ====================================================================================================================
#                                               PREDICTION + STATS ENGINES
# =====================================================================================================================
def quick_predict(
    resonance_ec: ECSystem,
    phi_predict: complex,
    observables: dict[str, Callable[[PredictPoint], complex]],
    resonance_dvr: DVR,
    resonance_energy: complex,
    bound_dvr: DVR | None=None,
    bound_energy: complex | None=None,
) -> dict[str, complex]:
    """
    Quickly predict many observables using EC. Returns dictionary of results
    where the keys share the same names as those in `observables` and the val is the computed observable.
    """
    phi_predict = complex(phi_predict)
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

    pt = PredictPoint(
        predicted_resonance_energy, predicted_resonance_state, phi_predict, copy.copy(resonance_dvr),
        bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None, 
        bound_energy=benergy, bound_state=bstate
    )
    for name, fn in observables.items():
        results[name] = fn(pt)
    return results


# =======================================================================================================
#                                            HELPERS
# =======================================================================================================
def residual(reference: complex, prediction: complex | np.ndarray) -> complex | np.ndarray:
    """ 
    Compute the complex-component-wise residual between `prediction` and `reference`. 
    If `prediction` is an array, the reference will be broadcast against the entire array,
    and the result will be the residual at each element of the array.
    """
    diff = prediction - reference
    return np.abs(np.real(diff)) + 1j*np.abs(np.imag(diff))
    
def relative_residual(reference: complex, prediction: complex | np.ndarray, tol: float=1e-9) -> complex:
    """ Compute the complex-component-wise relative residual between `prediction` and `reference`. """
    diff = prediction - reference
    if (np.abs(reference.real) < tol) or (np.abs(reference.imag) < tol):
        raise ValueError(
            f"Reference can't be zero in the real or imaginary component for relative residual. "
            f"Got a real or imaginary component of 0 within tolerance {tol:e}. Use `_residual` "
            f"instead."
        )
    return np.abs(np.real(diff)) / reference.real + 1j*np.abs(np.imag(diff)) / reference.imag

def quick_residuals(
    reference: dict[str, complex],
    prediction: dict[str, complex],
    relative: bool=False
) -> dict[str, complex]:
    resids = {}
    for name, val in prediction.items():
        r = reference[name]
        resids[name] = relative_residual(r, val) if relative else residual(r, val)
    return resids