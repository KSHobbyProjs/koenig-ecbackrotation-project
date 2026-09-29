"""
predict_utils.py

A set of convenience methods for getting predictions.
"""

import numpy as np

from ..dvr import DVR
from ..ec import ECSystem
from .sweep_utils import ScanPoint
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

# ====================================================================================================================
#                                               PREDICTION + STATS ENGINES
# =====================================================================================================================
def residual(reference: complex, prediction: complex) -> complex:
    """ Compute the complex-component-wise residual between `prediction` and `reference`. """
    diff = prediction - reference
    return np.abs(diff.real) + 1j*np.abs(diff.imag)
    
def relative_residual(reference: complex, prediction: complex, tol: float=1e-9) -> complex:
    """ Compute the complex-component-wise relative residual between `prediction` and `reference`. """
    diff = prediction - reference
    if (np.abs(reference.real) < tol) or (np.abs(reference.imag) < tol):
        raise ValueError(
            f"Reference can't be zero in the real or imaginary component for relative residual. "
            f"Got a real or imaginary component of 0 within tolerance {tol:e}. Use `_residual` "
            f"instead."
        )
    return np.abs(diff.real) / reference.real + 1j*np.abs(diff.imag) / reference.imag

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

def quick_predict(
    resonance_ec: ECSystem,
    phi_predict: complex,
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
        results["bound_energy"] = benergy

    pt = PredictPoint(
        predicted_resonance_energy, predicted_resonance_state, phi_predict, copy.copy(resonance_ec),
        bound_dvr=copy.copy(bound_dvr) if bound_dvr is not None else None, 
        bound_energy=benergy, bound_state=bstate
    )
    for name, fn in observables.items():
        results[name] = fn(pt)
    return results

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
