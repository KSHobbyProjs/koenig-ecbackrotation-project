"""
reference_utils.py

A few functions to help approximate the exact values. 
"""
import numpy as np

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
