# -*- coding: utf-8 -*-
"""
tikhonov.py

See docs/tikhonov.md for more information.
"""
from .dvr import DVRCore

import numpy as np
from numpy import pi, exp, sqrt

from typing import Callable

class Tikhonov:
    def __init__(self, n_dvr: int, L: float, r0=1.0, N=512, xmax=8.0):
        """
        Class for computing radial density using Tikhonov regularization
        DVRsystem : DVR
            The DVR system to compute the regularized state with respect to.
        r0 : float
            Reference scale for Tikhonov. Default 1.0.
        N : int
            Number of grid points for Fourier transform in Tikhonov. Default 512.
        xmax : float
            Maximum length to integrate to in log-space of Fourier transform. Default 8.0.
        kappa : float
            Tikhonov cutoff parameter.
        """
        self.n_dvr = int(n_dvr)
        self.L = float(L)
        self.dvrcore = DVRCore(n_dvr, L)
        
        self.r0 = float(r0)
        self.N = int(N)

        self.xmax = xmax
        self.xmin = -np.log(self.L / self.r0)
        self.x = np.linspace(self.xmin, self.xmax, self.N)

        self.r_log = self.r0 * exp(-self.x)    

        # dictionary for the possible tikhonov variables available as keys
        self.filter_dict = {
            "low_pass": Tikhonov.low_pass,
            "high_pass": Tikhonov.high_pass,
            "low_symmetric_pass": Tikhonov.low_symmetric_pass,
            "sigmoid_filter": Tikhonov.low_symmetric_pass,
        }
        
    # ---------------------------------------- TIKHONOV FILTERS --------------------------------------------
    @staticmethod
    def low_pass(rotation_angle, xis, kappa=1e-4):
        return 1.0 / (1.0 + kappa * exp(2.0*rotation_angle*xis))


    @staticmethod
    def high_pass(rotation_angle, xis, kappa=1e-4):
        return 1.0 / (1.0 + kappa * exp(-2.0*rotation_angle*xis))

    @staticmethod
    def low_symmetric_pass(rotation_angle, xis, kappa=1e-4):
        return 1.0 / (1.0 + kappa * exp(2.0*rotation_angle*np.abs(xis)))
        
    @staticmethod
    def sigmoid_filter(rotation_angle, xis, xi_minus=-50, xi_plus=50, width=1.0):
        # ~1 inside (xi_minus, xi_plus), smoothly -> 0 outside, over scale `width`
        left = 1.0 / (1.0 + exp(-(xis - xi_minus) / width))
        right = 1.0 / (1.0 + exp((xis - xi_plus) / width))
        return left * right

    # -------------------------------- CORE TIKHONOV METHODS ------------------------------
    def dvrstate_on_logmesh(self, dvr_state):
        """ Compute a dvr_state on the Tikhonov log mesh """
        return self.r_log, self.dvrcore.DVR_to_position_space(dvr_state, self.r_log)        

    def compute_fft(self, func):
        """ Compute the Fourier transform of the log-spaced wavefunction """
        # compute FFT
        # np.fft gives discrete FFT. Need dx / sqrt(2pi) to approx integral
        dx = self.x[1] - self.x[0]
        f_hat = np.fft.fft(func) * (dx / sqrt(2*pi))
        
        # compute frequency points (multiply by 2pi to turn cycle freq to ang freq)
        xis = np.fft.fftfreq(self.N, d=dx) * 2*pi
        return xis, f_hat

    def apply_tikhonov_filter(self, rotation_angle, xis, fftfunc, tikhonov_func="high_pass", *tfunc_args, **tfunc_kwargs):
        
        """ Apply the Tikhonov filter to the log-spaced fourier transform """
        """
        `tikhonov` below should be set to a filter that kills off your
        non-physical frequency information as checked by `backrotated_fft`
        """
        # apply Tikhonov and compute inverse FFT
        dxi = xis[1] - xis[0]

        tikhonov_func = self.filter_dict.get(tikhonov_func)
        if tikhonov_func is None:
            raise KeyError(f"tikhonov_func = {tikhonov_func} does not exist.\n"
                           f"Use a filter from the self.filter_dict dictionary."
                        )
        else:
            tikhonov = tikhonov_func(rotation_angle, xis, *tfunc_args, **tfunc_kwargs)
        # multiply by N bc numpy ifft has 1/N factor that numpy fft doesn't
        f_reg = np.fft.ifft(fftfunc * tikhonov * exp(rotation_angle*xis)) * (self.N*dxi/sqrt(2*pi))
        return f_reg
        
    # ----------------------------------------- TIKHONOV CONVENIENCE METHODS -----------------------------------   
    def backrotated_fft(self, func, rotation_angle):
        """ A method for checking the frequency domain of the back-rotated integrand """
        xis, f_hat = self.compute_fft(func)
        return xis, f_hat * exp(rotation_angle*xis)
        
    def compute_regularized_state(self, dvr_state, rotation_angle):
        """ Compute the regularized back-rotated state """
        func = self.dvrstate_on_logmesh(dvr_state)
        xis, f_hat = self.compute_fft(func)
        f_reg = self.apply_tikhonov_filter(rotation_angle, xis, f_hat)
        return f_reg
                      














