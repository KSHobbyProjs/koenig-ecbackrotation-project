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

from typing import Callable

import logging
logger = logging.getLogger(__name__)

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
    i, model, seed, phi_range, points, k = args
    logger.debug(f"Training system {i}")
    
    rng = np.random.default_rng(seed)
    # can't use np.random.uniform because it reads from process-global state. Under fork, 
    # every worker starts a copy of the same parent state, so drawing from the global 
    # generator in each child would silently produce correlated training angles
    phis = rng.uniform(phi_range[0], phi_range[1], points)
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
  def train(self, phi_range, points, k=1, n_workers=None):
    phi_range = np.asarray(phi_range)
    points = int(points)
    if phi_range.ndim != 1 or len(phi_range) != 2:
        raise ValueError(f"train expects to receive phi_range as a 1D list of the form [min, max], got shape {phi_range.shape}.")
        
    # get the log level set at the parent notebook so we can trickle it down to the workers
    current_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Training {self.n_ecsystems} EC systems.") 
      
    # generate independent seeds for each process
    seeds = np.random.SeedSequence().spawn(self.n_ecsystems)

    tasks = [(i, model, seed, phi_range, points, k)
             for i, (model, seed) in enumerate(zip(self.models, seeds))]

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
  def predict_DVR_states_at(self, phi_predict: float, n_workers=None):
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
    phi_predict = float(phi_predict)
    current_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Predicting DVR states for {self.n_ecsystems} EC systems.")
      
    tasks = [(i, model, phi_predict) for i, model in enumerate(self.models)]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(current_level,)) as ex:
        results = list(ex.map(_predict_DVR_states_worker, tasks))
    predicted_energies = np.array([r[0] for r in results])
    predicted_DVR_states = np.array([r[1] for r in results]).T
    return predicted_energies, predicted_DVR_states

  # ----------------------------------- PREDICTION CONVENIENCE CLASSES ---------------------------------
  def predict_energies_rrs(self, phi_predict: float, rotate_rr: bool=True, n_workers=None):
    """ Predict energies and rrs at phi_predict for all EC systems. """
    phi_predict = float(phi_predict)
    current_level = logging.getLogger().getEffectiveLevel()
    logger.info(f"Predicting energies and r^2 for {self.n_ecsystems} EC systems.")
      
    energies, states = self.predict_DVR_states_at(phi_predict, n_workers)
    rrs = np.array([self.system.compute_rr(state, rotate_rr=rotate_rr) for state in states.T])
    return energies, rrs

  def predict_densities_at(self, phi_predict: float, x_plot: np.ndarray | None=None, n_workers=None):
      """ Predict density at phi_predict for all EC systems. Shape (n_ecsystems, len(x_plot)). """
      phi_predict=float(phi_predict)
      current_level = logging.getLogger().getEffectiveLevel()
      logger.info(f"Predicting densities for {self.n_ecsystems} EC systems.")
      
      _, states = self.predict_DVR_states_at(phi_predict, n_workers=n_workers)
      densities = np.array([self.system.compute_density(state, x_plot) for state in states.T])
      return densities
      
  def predict_energies_rrs_stats(self, phi_predict: float, rotate_rr: bool=True, n_workers=None):
    """ Gets energy stats and rrs stats at `phi_predict`. See `compute_stats` for output shapes. """
    phi_predict=float(phi_predict)
      
    energies, rrs = self.predict_energies_rrs(phi_predict, rotate_rr, n_workers=n_workers) 
    energies_stats = ECEnsemble.compute_stats(energies)
    rrs_stats = ECEnsemble.compute_stats(rrs)
    return energies_stats, rrs_stats

  # technically, ECEnsemble can actually compute the reference energy, rr, and density since it has access to the
  # dvr system. However, we require the references as input to avoid computing errors relative to possibly flawed data.
  def predict_energies_rrs_resid_stats(self, reference_energy, reference_rr, phi_predict, rotate_rr, n_workers=None):
      """ 
      Computes energy residual stats and rrs residual stats at `phi_predict`. 
      See `compute_resid_stats` for output shapes. 
      """
      reference_energy = complex(reference_energy)
      reference_rr = complex(reference_rr)
      phi_predict=float(phi_predict)
      
      energies, rrs = self.predict_energies_rrs(phi_predict, rotate_rr, n_workers=n_workers)
      energies_resid_stats = ECEnsemble.compute_resid_stats(reference_energy, energies)
      rrs_resid_stats = ECEnsemble.compute_resid_stats(reference_rr, rrs)
      return energies_resid_stats, rrs_resid_stats

  def predict_density_stats(self, phi_predict: float, x_plot: np.ndarray | None=None, n_workers=None):
      """ Gets density stats at `phi_predict`. See `compute_stats` for output shapes. """
      phi_predict = float(phi_predict)
      x_plot = np.asarray(x_plot)
      if x_plot.ndim != 1: 
          raise ValueError(f"predict_density_stats expects a 1D array of x_plot values, got {x_plot.shape}.")
      
      densities = self.predict_densities_at(phi_predict, x_plot, n_workers=n_workers)
      
      return ECEnsemble.compute_stats(densities, axis=0)

  def predict_density_resid_stats(self, x_plot, reference_density, phi_predict, n_workers=None):
      """
      Computes density residual stats at `phi_predict`. See `compute_resid_stats` for output shapes.
      `x_plot` should be the same `x_plot` that `reference_density` is computed on.
      """
      phi_predict = float(phi_predict)
      x_plot = np.asarray(x_plot)
      reference_density = np.asarray(reference_density)
      if x_plot.ndim != 1 or reference_density.ndim != 1 or len(x_plot) != len(reference_data): 
          raise ValueError(f"predict_density_stats expects a 1D array of x plot values and a 1D array of "
                           f"reference density values of the same length, got {x_plot.shape} vs {reference_density.shape}.")
      
      densities = self.predict_densities_at(phi_predict, x_plot, n_workers=n_workers)
      return ECEnsemble.compute_resid_stats(reference_density, densities, axis=0)

      
      
  # ------------------------------------ UTILITY METHODS ------------------------------------------------
  @staticmethod
  def compute_resid_stats(reference: complex | np.ndarray, data: np.ndarray, axis: int | None=None):
    """
    Computes the residual of the data along with the stats associated with the residual.

    Parameters
    ----------
    reference: complex | np.ndarray
        The reference value(s) to take the residual with respect to. Must broadcast
        against `data` with the `axis` dimension removed; i.e., against the shape `median`
        would have (see Returns below). A scalar applies the same reference to every point;
        an array supplies a different reference per point.
    data: np.ndarray
        The data to take residuals of. Arbitrary shape.
    axis: int
        The axis along which to compute the stats. Default is None, in which case stats are 
        taken along the flattened version of `data`. 

    Returns
    -------
    median: np.ndarray or complex
        The median of the residual data set.
    err68: np.array([np.ndarray, np.ndarray])
        The minus and plus 68%-percentile error bands for the residual data set.
    err95: np.array([np.ndarray, np.ndarray])
        The minus and plus 95%-percentile error bands of the residual data set.
    """
    reference = np.asarray(reference)
    data = np.asarray(data)

    if np.any(reference.imag == 0):
        raise ValueError(
            "Reference has zero imaginary part. Im-relative-residual is undefined "
            "for a purely real reference. Use Re-only stats instead."
        )
    diff = data - reference
    resids = diff.real/reference.real + 1j*diff.imag/reference.imag
    return ECEnsemble.compute_stats(resids, axis=axis)
    
      
  @staticmethod
  def compute_stats(data: np.ndarray, axis: int=0):
    """
    Computes the median, 68%-percentile error bands, and 95%-percentile error bands for data. Assumes the 
    data is given as an 1D array of complex numbers or some ndarray of complex numbers.
    Parameters
    ----------
    data: np.ndarray
        Data to take stats of. Arbitrary shape of data. 
    axis: int or None.
        The axis along which to compute the stats. Default is None, in which case stats are taken along
        the flattened version of `data`.
    
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
      
    median = np.median(data.real, axis=axis) + np.median(data.imag, axis=axis)*1j
    err68 = ECEnsemble.compute_percentile_error_bands(data, 68.2, axis=axis)
    err95 = ECEnsemble.compute_percentile_error_bands(data, 95.4, axis=axis)
    return median, err68, err95

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
      
    median = np.median(data.real, axis=axis) + np.median(data.imag, axis=axis)*1j
    err_p = (np.percentile(data.real, 50.+percentile/2, axis=axis) +
           np.percentile(data.imag, 50.+percentile/2, axis=axis)*1j) - median
    err_m = median - (np.percentile(data.real, 50.0-percentile/2, axis=axis) +
                      np.percentile(data.imag, 50.0-percentile/2, axis=axis)*1j)
    return np.array([err_m, err_p])