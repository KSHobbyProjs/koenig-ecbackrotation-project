# -*- coding: utf-8 -*-
"""
ecensemble.py

A class that handles running multiple EC systems in parallel and computing statistics.
"""
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor

from .ec import ECSystem, TakagiRegularization
from .dvr import DVR

import numpy as np
from scipy.stats import qmc

from typing import Callable
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------ HELPER FUNCTIONS --------------------------------------
@dataclass
class StatPoint:
    """
    Statistics for the data at a point.

    med corresponds to the median. err68 corresponds to the 68% percentile error bands: err68[0] is the 
    lower bound, err68[1] is the upper bound. err95 corresponds to the 95% percentile error bands.

    Generally, this corresponds to a single piece of data, like resonance energy or r^2, in which case
    med is complex and err68 / err95 are complex 2-tuples. However, it can also apply to objects like
    the density, in which case med is an array, corresponding to the median value of the density at every
    point along x_plot, and err68 / err95 are 2-tuples of arrays such that err68[0 (1), i] is the lower
    (upper) bound of the 68th percentile at x = x_plot[i]. 
    """
    med: complex | np.ndarray
    err68: np.array([complex, complex]) | np.array([np.ndarray, np.ndarray])
    err95: np.array([complex, complex]) | np.array([np.ndarray, np.ndarray])

def _latin_hypercube(d, rng):
    """ 
    Latin hypercube sampling. A Latin hypercube is a way of choosing n random points in a d-dimensional
    box so that every one-dimensional projection is evenly stratified. Ex: sampling from an area in the
    complex plane: with n sample points, each axis is cut into n strips, so the real axis is split into
    n vertical strips of width Delta(Re(theta))/n, with one point in each. Similarly for the imaginary axis.

    This doesn't completely discourage two points from being close in 2D. optimization="random-cd" 
    is added to discourage 2D clumping.

    Need a try, except because some versions of scipy use the argument "rng", some use "seed".
    """
    try:
        return qmc.LatinHypercube(d=d, rng=rng, optimization="random-cd")
    except TypeError:
        return qmc.LatinHypercube(d=d, seed=rng, optimization="random-cd")

def _sample_complex_phis(
    phi_real_range,
    phi_imag_range,
    phi_grid_points,
    phi_complex_points,
    ss_a,
    ss_b
):
    """
    Exactly one of phi_grid_points / phi_complex_points is non-None (validated in `train` below).
    tensor: nr random Re coordinates x ni random Im coordinates -> nr*ni points on a tensor grid. Corresponds to phi_grid_points.
    lhs: n points from a 2D Latin hypercube over the rectangle. Corresponds to phi_complex_points.
    """
    # if phi_grid_points is populated, produce a random tensor grid
    if phi_grid_points is not None:
        re = np.random.default_rng(ss_a).uniform(*phi_real_range, phi_grid_points[0])
        im = np.random.default_rng(ss_b).uniform(*phi_imag_range, phi_grid_points[1])
        return (re[:, None] + 1j*im[None, :]).ravel()

    # else: produce a Latin hypercube distribution
    u = _latin_hypercube(2, np.random.default_rng(ss_a)).random(phi_complex_points)
    pts = qmc.scale(u, [phi_real_range[0], phi_imag_range[0]],
                       [phi_real_range[1], phi_imag_range[1]])
    return pts[:, 0] + 1j*pts[:, 1]

# -------------------------------------- WORKER FUNCTIONS FOR PARALLEL -----------------------------
# ================================================================================================
# Module-level worker functions for parallelization
#
# These must live at module level. ProcessPoolExecutor ships tasks to worker processes via
# pickle, and pickle serializes a function by reference. A function defined inside a method
# has no such resolvable name, so pickling fails immediately. 
# =================================================================================================
def _init_worker(log_level):
    """
    Runs once the moment each worker process starts. Caps this proces's BLAS library (OpenBLAS, in
    my case) to a single thread. Without this, each worker process would try to multithread its own
    eig() calls internally. N processes * T BLAS threads can exceed core count, called "oversubscription",
    which makes the parallel version slower than serial (each process has to fight over threads).
    """
    threadpool_limits(limits=1)
    logging.basicConfig(level=log_level, format="[PID %(process)d] [%(levelname)s] %(message)s", force=True) # trickle logging from parent notebook to workers

def _train_worker(args):
    (i, model, seed, phi_real_range, phi_real_points,
     phi_imag_range, phi_grid_points, phi_complex_points, k) = args
    logger.debug(f"Training system {i}")

    ss_real, ss_ca, ss_cb = seed.spawn(3)

    parts = []
    if phi_real_points is not None:
        parts.append(np.random.default_rng(ss_real).uniform(*phi_real_range, phi_real_points).astype(np.complex128))
    if phi_imag_range is not None:
        parts.append(_sample_complex_phis(phi_real_range, phi_imag_range,
                                          phi_grid_points, phi_complex_points, ss_ca, ss_cb))
    phis = np.concatenate(parts)
    model.train(phis, k)
    return model # mutated model is pickled back to the main process

def _construct_basis_worker(args):
    i, model, augment, eps = args
    logger.debug(f"Constructing basis for system {i}")
    
    model.construct_basis(augment=augment, eps=eps)
    return model

def _predict_DVR_states_worker(args):
    i, model, phi_predict = args
    logger.debug(f"Predicting for system {i}")
    return model.predict_DVR_state_at(phi_predict)

# ---------------------------------------- EC ENSEMBLE CLASS -----------------------------------------------

class ECEnsemble:
  def __init__(self, 
               n_ecsystems: int, 
               DVRsystem: DVR, 
               resonance_energy: complex,
               regularizer: TakagiRegularization | None=None
            ):
    """
    Creates a setup to perform many EC simulations at once to produce stats and error bounds
    """
    self.system = DVRsystem
    self.n_ecsystems = n_ecsystems
    self.models = [ECSystem(DVRsystem, resonance_energy, regularizer) for _ in range(n_ecsystems)]

  # ----------------------------------- CORE ECENSEMBLE METHODS ---------------------------------------
  def clear_all(self):
    for model in self.models:
      model.clear_all()

  # ------------------------------------ Training ------------------------------------------------------
  def train(
      self, 
      phi_real_range: list[float, float], 
      phi_real_points: int | None, 
      phi_imag_range: list[float, float] | None=None, 
      phi_grid_points: list[int, int] | None=None, 
      k: int=1,
      *,
      phi_complex_points: int | None=None,
      seed: int | None=None,
      n_workers=None,
  ):
    """
    Train each EC system on a real-axis set plus (optionally) a complex set.

    Real-axis set: `phi_real_points` uniformally draws from `phi_real_range`. Set to None to omit.

    Complex set: requires `phi_imag_range` and only one of the following:
        `phi_grid_points`=[nr, ni] -> tensor grid: nr random Re coordinates in phi_real_range times
                                                   ni random Im coordinates in phi_imag_range, giving
                                                   nr*ni points.
        `phi_complex_points`=n     -> Latin hypercube: n points over the rectangle phi_real_range x
                                                       phi_imag_range.

    seed: Ensemble seed. With a fixed seed, ec system i's real-axis points are identical across repeated `train` 
    calls (system i is still random from system j, but each system's random distribution is identically replicated 
    if you call `train` again), regardless of which complex set, if any, is added.
    """
    # -------------------------------------------- VALIDATION --------------------------------------------------
    phi_real_range = np.asarray(phi_real_range, dtype=np.float64)
    if phi_real_range.shape != (2,) or not phi_real_range[0] < phi_real_range[1]:
        raise ValueError(f"phi_real_range must be [min, max] with min < max, bot {phi_real_range}")

    if phi_real_points is not None:
        phi_real_points = int(phi_real_points)
        if phi_real_points < 1:
            raise ValueError(f"phi_real_points must be positive or None, got {phi_real_points}")

    n_counts = (phi_grid_points is not None) + (phi_complex_points is not None)
    if n_counts == 2:
        raise ValueError("Give phi_grid_points (tensor) OR phi_complex_points (LHC).")
    if (phi_imag_range is None) != (n_counts == 0):
        raise ValueError("phi_imag_range must be given together with phi_grid_points or phi_complex_points")
    if phi_real_points is None and phi_imag_range is None:
        raise ValueError("train needs a real-axis set, a complex set, or both.")

    if phi_imag_range is not None:
        phi_imag_range = np.asarray(phi_imag_range, dtype=np.float64)
        if phi_imag_range.shape != (2,) or not phi_imag_range[0] < phi_imag_range[1]:
            raise ValueError(f"phi_imag_range must be [min, max] with min < max, got {phi_imag_range}.")
    if phi_grid_points is not None:
        phi_grid_points = np.asarray(phi_grid_points, dtype=np.int64)
        if phi_grid_points.shape != (2,) or np.any(phi_grid_points < 1):
            raise ValueError(f"phi_grid_points must be [nr, ni] with positive entries, got {phi_grid_points}.")
    if phi_complex_points is not None:
        phi_complex_points = int(phi_complex_points)
        if phi_complex_points < 1:
            raise ValueError(f"phi_complex_points must be positive, got {phi_complex_points}.")
    # --------------------------------------------------------------------------------------------------------------------------------------

      
    # get the log level set at the parent notebook so we can trickle it down to the workers
    current_level = logging.getLogger().getEffectiveLevel()
    scheme = "tensor" if phi_grid_points is not None else "lhc" if phi_complex_points is not None else "real-only"
    ss = np.random.SeedSequence(seed)
    self.seed_entropy = ss.entropy # pass this back as `seed` to reproduce the run exactly
    logger.info(f"Training {self.n_ecsystems} EC systems ({scheme}, seed entropy {ss.entropy}.") 
      
    # generate independent seeds for each process
    seeds = ss.spawn(self.n_ecsystems)
    tasks = [(i, model, s, phi_real_range, phi_real_points, 
              phi_imag_range, phi_grid_points, phi_complex_points, k)
             for i, (model, s) in enumerate(zip(self.models, seeds))]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(current_level,)) as ex:
        # map() sends one task per model to the pool, blocks until every one
        # has finished, and returns results in the same order that the tasks
        # were submitted, so this list lines back up with self.models positionally.
        self.models = list(ex.map(_train_worker, tasks))

  # ---------------------------------- Construct Basis -------------------------------------------------
  def construct_bases(self, augment=None, eps: float=1.0e-8, n_workers=None):
    current_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Constructing bases for {self.n_ecsystems} EC systems.")
      
    tasks = [(i, model, augment, eps) for i, model in enumerate(self.models)]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(current_level,)) as ex:
        self.models = list(ex.map(_construct_basis_worker, tasks))

  # ---------------------------------- Predict ---------------------------------------------------------
  def predict_DVR_states_at(self, phi_predict: complex, n_workers=None):
    """
    Gets the predicted energy and DVR state at phi_predict for each model

    Parameters
    ----------
    phi_predict: float
        The complex-scaling rotation angle to predict at.

    Returns
    -------
    predicted_energies: np.ndarray
        The predicted resonance energies for each model. Shape (n_ecsystems)
    predicted_DVR_states: np.ndarray
        The predicted DVR state of the resonance for each model. Shape (num_dvr_points, n_ecsystems). 
    """
    phi_predict = complex(phi_predict)
    current_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Predicting DVR states for {self.n_ecsystems} EC systems.")
      
    tasks = [(i, model, phi_predict) for i, model in enumerate(self.models)]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(current_level,)) as ex:
        results = list(ex.map(_predict_DVR_states_worker, tasks))
    predicted_energies = np.array([r[0] for r in results])
    predicted_DVR_states = np.array([r[1] for r in results]).T
    return predicted_energies, predicted_DVR_states

  # ------------------------------------ UTILITY METHODS ------------------------------------------------      
  @staticmethod
  def compute_stats(data: np.ndarray, axis: int=0):
    """
    Computes the median, 68%-percentile error bands, and 95%-percentile error bands for data. Assumes the 
    data is given as an 1D array of complex numbers or some ndarray of complex numbers.
    Parameters
    ----------
    data: np.ndarray
        Data to take stats of. Arbitrary shape of data. 
    axis: int
        The axis along which to compute the stats. Default is 0.
    
    Returns
    -------
    median: np.ndarray or complex
        The median of the data set. If data is more than 1-dimensional, then median is an array with the same
        shape as data except for the dimension corresponding to axis flattened. For instance, if data is shape (3, 3),
        and axis=1, then median has shape (3).
    err68: np.array([np.ndarray, np.ndarray]) or np.array([complex, complex])
        The minus and plus 68%-percentile error bands. err68[0] corresponds to the minus; err68[1] corresponds to the plus. 
        Same shape logic as for median.
    err95: np.array([np.ndarray, np.ndarray]) or np.array([complex, complex])
        The minus and plus 95%-percentile error bands. err95[0] corresponds to the minus; err95[1] corresponds to the plus.
        Same shape logic as for median.
    """
    data = np.asarray(data)
      
    median = np.median(np.real(data), axis=axis) + np.median(np.imag(data), axis=axis)*1j
    err68 = ECEnsemble.compute_percentile_error_bands(data, 68.2, axis=axis)
    err95 = ECEnsemble.compute_percentile_error_bands(data, 95.4, axis=axis)
    return StatPoint(
        med=median,
        err68=err68,
        err95=err95
    )

  @staticmethod
  def compute_percentile_error_bands(data: np.ndarray, percentile: float, axis: int | None=None):
    """
    Computes the percentile error bands of a set of complex-valued data.

    Parameters
    ----------
    data: np.ndarray
        Data to take stats of. Shape (len(data)).
    percentile: float
        Percentile range to take error bounds with respect to.
    axis: int or None.
        The axis along which to compute the stats. Default is None, in which case stats are taken along
        the flattened version of `data`.

    Returns
    -------
    err: np.array([np.ndarray, np.ndarray]) or np.array([complex, complex])
        The minus and plus percentile error bands. First element corresponds to minus; second element corresponds to plus.
        Same shape logic as discussed in `compute_stats`.
    """
    # compute median (technically un-needed if we compute it before, but quick enough that it doesn't matter
    data = np.asarray(data)
      
    median = np.median(np.real(data), axis=axis) + np.median(np.imag(data), axis=axis)*1j
    err_p = (np.percentile(np.real(data), 50.+percentile/2, axis=axis) +
           np.percentile(np.imag(data), 50.+percentile/2, axis=axis)*1j) - median
    err_m = median - (np.percentile(np.real(data), 50.0-percentile/2, axis=axis) +
                      np.percentile(np.imag(data), 50.0-percentile/2, axis=axis)*1j)
    return np.array([err_m, err_p])