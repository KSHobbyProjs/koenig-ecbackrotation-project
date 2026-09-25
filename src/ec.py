# -*- coding: utf-8 -*-
"""
ec.py

Module to store
"""
from .dvr import DVR

import numpy as np
from numpy import exp, sqrt
import scipy.linalg as sl

from typing import Callable

import logging
logger = logging.getLogger(__name__)

class TakagiRegularization:
    """
    EC overlap matrix under the c-product, S, is square, complex symmetric, so it has
    a Takagi decomposition S = U D U^T, U unitary and D real, positive, diagonal.

    Elements of D are square roots of eigenvalues of S^+ S (singular values of S).

    Since S is the overlap matrix, directions with singular values near 0 represent 
    linear dependence. Killing off these directions results in a less singular matrix,
    and it turns the generalized eigenvalue problem into a normal one.
    """
    def __init__(self, cutoff: float=1.0e-5, tol: float=1.0e-8):
        self.cutoff, self.tol = float(cutoff), float(tol)

        self._cached_P = None
        self._cached_N_mat = None

    def __call__(self, N_mat, H):
        """
        Projects the eigenvalue problem H

        Parameters
        ----------
        N_mat : np.ndarray
            The overlap matrix of the EC eigenvectors. Shape (n, n) (for some EC eigenvectors).
        H : np.ndarray
            The EC projected Hamiltonian matrix. Shape (n, n) (for n EC eigenvectors).
        """
        N_mat = np.asarray(N_mat)
        H = np.asarray(H)

        if N_mat.ndim != 2 or H.ndim != 2:
            raise ValueError(f"TakagiRegularization.__call__ requires N_mat and H to be 2D arrays, got "
                             f"{N_mat.shape} and {H.shape}.")
        
        P = self.get_P(N_mat)                 # get projector onto reduced subspace
        H_reduced = P.T @ H @ P                # project H onto reduced space (N = I in reduced space)
        eigs, eigvecs = np.linalg.eig(H_reduced)
        original_eigvecs = P @ eigvecs
        return eigs, original_eigvecs
    
    def get_P(self, N_mat):
        """
        Computes the Takagi decomp of a matrix N_mat, truncates off all
        vectors corresponding to an eigenvalue less than cutoff
        and returns the projector P = U^* D^{-1/2} (U^* is conj not adjoint).
        """
        N_mat = np.asarray(N_mat)
        if N_mat.ndim != 2:
            raise ValueError(f"get_P requires N_mat to be a 2D array, got {N_mat.shape}.")
        
        if self._cached_P is not None and np.array_equal(N_mat, self._cached_N_mat):
            return self._cached_P
            
        # make Hermitian matrix
        M = N_mat @ N_mat.conj().T
        DD, U = np.linalg.eigh(M)
        
        # kill off ghost directions (directions representing linear dependence of the basis formed by mat)
        DD = np.clip(DD.real, 0, None) # probably not necessary, but guards against negative roundoff noise before sqrt
        mask = sqrt(np.abs(DD)) > np.max(sqrt(np.abs(DD))) * self.cutoff 
        D = sqrt(DD[mask])
        U = U[:, mask].conj()
        
        # Takagi phase-fixing (otherwise, eigenvectors in U can vary by arbitrary phase)
        diag_vals = np.array([U[:,i] @ N_mat @ U[:,i] for i in range(U.shape[1])])
        phases = np.angle(diag_vals)
        U = U * exp(-1j*phases/2)[None, :]
        
        Dinv = np.diag(1 / D)
        P = U @ Dinv**.5

        self._cached_P = P
        self._cached_N_mat = N_mat.copy()
        
        logger.debug(f"Takagi kept {P.shape[1]} / {N_mat.shape[0]} directions; "
                     f"cond(P^T N P) = {np.linalg.cond(P.T @ N_mat @ P):.3f}")
        # testing if P was produced correctly: P^T N P = I
        resid = np.allclose(P.T @ N_mat @ P, np.eye(P.shape[1]), rtol=self.tol)
        diff = np.max(np.abs(P.T @ N_mat @ P - np.eye(P.shape[1])))
        if not resid: logger.warning(f"P^T N P not identity within tol {self.tol:e}. Max difference: {diff}")
        return P
        
        # if M is degenerate, not every orthonormal basis satisfies N = U D U^T
        # check if P^T N P = I (it should unless M is degenerate)
        # TODO: catch if M is degenerate, and simult diag to select 
        #       the correct orthonorm basis s.t. P.T N P = I
        
class ECSystem:
  def __init__(self,
               DVRsystem: DVR, 
               resonance_energy: complex,
               regularizer: TakagiRegularization | None=None
            ):
    """
    Initializes an eigenvector continuation setup that handles complex scaling.

    Parameters
    ----------
    DVRsystem: DVR
        DVR instance.
    resonance_energy: complex
        Complex resonance energy.
    regularizer: TakagiRegularization | None
        Regularizer for making the overlap matrix less singular.
    """
    self.system = DVRsystem
    self.resonance_energy = complex(resonance_energy)
    self.regularizer = regularizer 

    self.training_eigvals = []
    self.training_eigvecs = []

  # =====================================================================================================
  #                                           CORE EC SETUP
  # =====================================================================================================
  def clear_all(self):
    self.training_eigvals = []
    self.training_eigvecs = []

  # -----------------------------------------------------------------------------------------------------
  # Training 
  # ------------------------------------------------------------------------------------------------------
  def train(self, rotation_angles: np.ndarray, k: int=1):
    # sampling eigenvectors corresponding to specific resonance state for many phi
    rotation_angles = np.asarray(rotation_angles)
    if rotation_angles.ndim != 1:
        raise ValueError(f"train expects a 1D array of rotation angles, got {rotation_angles.shape}.")
      
    for phi in rotation_angles:
      phi = complex(phi)
      self.system.reset_rotation_angle(phi)
      eigval, eigstate = self.system.closest_to_resonance(self.resonance_energy, k)
      for i, val in enumerate(eigval):
        self.training_eigvals.append(val)
        self.training_eigvecs.append(eigstate[:, i])

  def train_from_outside_data(self, rotation_angles: np.ndarray, states: np.ndarray):
      """
      Method that imports training eigvals from outside data.

      Assumes that the training states are c-normalized.

      Input
      -----
      rotation_angles : array-like
          List of rotation_angles corresponding to each state in `states`.
      states : ndarray
          Array of states. Shape (self.n, num_states)
      """
      rotation_angles = np.asarray(rotation_angles)
      states = np.asarray(states)
      if rotation_angles.ndim != 1:
          raise ValueError(f"train_from_outside_data expects a 1D array of rotation angles, got {rotation_angles.shape}.")
      if states.ndim != 2:
          raise ValueError(f"train_from_outside_data expects a 2D array of states, got {states.shape}.")
      if rotation_angles.shape[0] != states.shape[1]:
          raise ValueError(f"rotation_angles and states need to correspond. "
                           f" Got {rotation_angles.shape[0]} vs {states.shape[1]}."
                          )
      
      for i, phi in enumerate (rotation_angles):
          phi = complex(phi)
          self.system.reset_rotation_angle(phi)
          self.training_eigvecs.append(states[:, i])

          # find training eigvals via (statei | H(theta) | statei) = E
          self.training_eigvals.append(np.sum(states[:, i] * self.system.H @ states[:, i]))
  
  # ----------------------------------------------------------------------------------------------------------------
  # Construct basis
  # ----------------------------------------------------------------------------------------------------------------
  def construct_basis(self, augment: str | None=None, eps: float=1e-8):
    # constructing EC basis with c-product
    if augment == 'conj':
        BA_vecs = [np.conj(vec) for vec in self.training_eigvecs]
        self.training_eigvecs.extend(BA_vecs)
    self.EC_basis = np.transpose(np.vstack(self.training_eigvecs))
    self.EC_basis_T = np.transpose(self.EC_basis)
    self.N_mat = self.EC_basis_T @ self.EC_basis

    # Tikhonov regularization (not the same as the back-rotation reg)
    self.N_mat = self.N_mat + eps * np.eye(self.N_mat.shape[0])
    cond = np.linalg.cond(self.N_mat)
    if cond > 1e6:
      logger.debug(f"N matrix nearly singular: cond(N) = {cond:e}.")

    # if there's a regularizer, construct the reduced matrix P now
    if self.regularizer is not None:
        self.regularizer.get_P(self.N_mat)

  # ----------------------------------------------------------------------------------------------------------------
  # Predict
  # ---------------------------------------------------------------------------------------------------------------
  def predict_EC_state_at(self, rotation_angle_predict: complex):
      """
      Predict EC eigenfunctions and eigenvalues at a given rotation angle.

      Parameters
      ----------
      rotation_angle_predict: float
          The rotation angle EC predicts at.
      truncate: bool, optional
          Whether or not to truncate off ghost directions. Default is True.

      Returns
      -------
      (eigvals, eigvecs) : (np.ndarray, np.ndarray)
          eigvals and EC eigvecs at given rotation angle. 
          eigvals shape (n_eigvals,), eigvecs shape (n_ec, n_eigvals)
      """
      rotation_angle_predict = complex(rotation_angle_predict)
      self.system.reset_rotation_angle(rotation_angle_predict)
      H_target = self.system.H
      H_proj = self.EC_basis.T @ H_target @ self.EC_basis

      # test if there are ghost directions due to singularity of N matrix
      # and cut them off before solving 
      # this is unecessary except in situations where a near-singular N 
      # matrix is unavoidable
      if self.regularizer is not None:
          return self.regularizer(self.N_mat, H_proj)
      return sl.eig(H_proj, b=self.N_mat)
      
  def predict_DVR_state_at(self, rotation_angle_predict: complex):
      """
      Use EC to predict the DVR state with the energy most like the resonance energy.

      Returns
      -------
      energy: complex
          Predicted resonance energy.
      DVRstate: np.ndarray
          Predicted DVR state. Shape (num_dvr_point).
      """
      rotation_angle_predict = complex(rotation_angle_predict)
      eigvals, eigvecs = self.predict_EC_state_at(rotation_angle_predict)
      
      # cannot uniquely determine which eigvec corresponds to
      # resonance when phi = 0, so we forcefully find it
      # (this is contrived but all we can do)
      energy_idx = np.nanargmin(np.abs(self.resonance_energy - eigvals))
      DVRstate = self.reconstruct_DVRstate_from_ECState(eigvecs[:, energy_idx])
      return eigvals[energy_idx], DVRstate
  
  # =================================================================================================================
  #                                   CONVENIENCE PREDICTION METHODS 
  # ===================================================================================================================
  def predict_energies_rrs(self, rotation_angles_predict: np.ndarray, rotate_rr: bool=True):
    """
    Computes <r^2> and E for resonance for rotation angles in rotation_angles_predict 

    Parameters
    ----------
    rotation_angles_predict : ndarray or array-like
        Array of rotation angles to compute <r^2> and E at.
    rotate_rr (optional): bool, default False
        Decided if r^2 operator should be complex-rotated (True) or not (False).
        
    Returns
    -------
    rrs : ndarray
        Array of <r^2> data.
    energies : ndarray
        Array of energy data.

    Notes
    -----
    EC can't uniquely determine which eigvec corresponds to the resonance as we adjust phi,
    so we forcefully find it by choosing the energy that best matches the known resonance freq.
    """
    rotation_angles_predict = np.asarray(rotation_angles_predict)
    if rotation_angles_predict.ndim != 1:
      raise ValueError(f"predict_energies_rrs expects a 1D array of rotation angles, got {rotation_angles_predict.shape}.")
          
    energies, rrs = [], []
    for phi in rotation_angles_predict:
      phi = complex(phi)
      energy, eigstateDVR = self.predict_DVR_state_at(phi)
      energies.append(energy)
        
      rr = self.system.compute_rr(eigstateDVR, rotate_rr=rotate_rr, regulator=False)
      rrs.append(rr)
    return energies, rrs

  def predict_density_at(self, rotation_angle_predict: complex, x_plot: np.ndarray | None=None):
    """
    Returns
    -------
    density: np.ndarray
        The density as a function of position. Shape (len(x_plot)).
    """
    rotation_angle_predict = complex(rotation_angle_predict)
    dvr_state = self.predict_DVR_state_at(rotation_angle_predict)[1]
    return self.system.compute_density(dvr_state, x_plot)

  # ================================================================================================================
  #                                        UTILITY METHODS
  # =================================================================================================================
  def reconstruct_DVRstate_from_ECState(self, EC_state):
    """ 
    Reconstruct a DVR state from the EC state using the EC training vectors.

    Automatically c-normalizes.

    Parameters
    ----------
    EC_state: np.ndarray
        States in EC basis. Shape (num_training_vecs).

    Returns
    -------
    np.ndarray
        DVR state corresponding to EC_state. Shape (num_dvr_points).
    """
    EC_state = np.asarray(EC_state)
    if EC_state.ndim != 1:
        raise ValueError(f"reconstruct_DVRstate_from_ECState expects a 1D EC state, got {EC_state.shape}.")
    state = EC_state @ np.array(self.training_eigvecs)
    return DVR.cnormalize(state)
