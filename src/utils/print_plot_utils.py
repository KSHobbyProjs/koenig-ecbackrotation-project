"""
utils.py
"""

import numpy as np
from pathlib import Path

from typing import Callable
from .sweep_utils import ScanPoint
from ..dvr import DVR
from ..ec import ECSystem
from ..ecensemble import StatPoint

import matplotlib.pyplot as plt

# ===============================================================================================================
#                                  PRINTING AND PLOTTING HELPERS
# ===============================================================================================================
# ---------------------------------------------------------------------------------------------------------------
# Printing utils
# ---------------------------------------------------------------------------------------------------------------  
def print_training_data(results: dict[str, np.ndarray]):
    for name, result in results.items():
        print(
            f"Training {name}: "
            f"({np.median(np.real(result)):.5f} +- {np.std(np.real(result)):.5f}) "
            f"+ i({np.median(np.imag(result)):.5f} +- {np.std(np.imag(result)):.5f})"
        )

def print_quick_predict(results: dict[str, complex]):
    for name, result in results.items():
        print(f"{name}:{result.real:.5f} + i{result.imag:.5f}")

def print_quick_predict_ensemblestats(results: dict[str, StatPoint]):
    for name, result in results.items():
        mr, mi = np.real(result.med), np.imag(result.med)
        err68r, err68i = np.real(result.err68), np.imag(result.err68)
        err95r, err95i = np.real(result.err95), np.imag(result.err95)
        print(
            f"{name}: "
            f"Median: {mr:.5f} + i{mi:.5f}. "
            f"68%: [{mr-err68r[0]:.5f}, {mr+err68r[1]:.5f}] + i[{mi-err68i[0]:.5f}, {mi+err68i[1]:.5f}]. "
            f"95%: [{mr-err95r[0]:.5f}, {mr+err95r[1]:.5f}] + i[{mi-err95i[0]:.5f}, {mi+err95i[1]:.5f}]."
        )
    
#----------------------------------------------------------------------------------------------------------------
# Plotting utils
# ---------------------------------------------------------------------------------------------------------------    
def get_plots_path():
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "plots"
    
def plot_stats(fig, axs, xs, med, err68, err95, cutoff: int=0, color='red'):
    """ 
    Plots EC ensemble stats with error bands. Designed to plot data and error bars as number of training
    points varies, but `xs`, `med`, `err68`, and `err95` can be anything.

    Parameters
    ----------
    fig: plt.fig 
        the matplotlib figure.
    axs: tuple[plt.ax, plt.ax]
        Pair of matplotlib axes, one for the real component; one for the imaginary component.
    xs: np.ndarray
        The x-axis data. Shape (len(xs),).
    med: np.ndarray
        The median of whatever data set as x varies. Shape (len(xs),).
    err68: np.ndarray
        The 68% percentile error bands corresponding to the data in `med`. Shape (len(xs), 2). err68[:, 0]
        is the minimum for the error band as x varies, and err68[:, 1] is the maximum for the error band as 
        x varies.
    err95: np.ndarray
        Exactly as `err68` except for the 95% percentile error bands.
    cutoff: int
        Cuts off the first "cutoff" data points. For example, if cutoff=1, then only xs[1:], med[1:], etc. will be plotted. Default is no cutoff.
    color: str
        Color of the data. Default is red.
    """
    ax1, ax2 = axs
    xs, med, err68, err95 = xs[cutoff:], med[cutoff:], err68[cutoff:, :], err95[cutoff:, :]
    
    # real component 
    ax1.scatter(xs, np.real(med), color=color, marker='x', label='Predict (median)')
    ax1.errorbar(
        xs, np.real(med),
        yerr=np.real(err68.T),
        fmt='none',
        ecolor=color, 
        elinewidth=2.0, 
        alpha=1.0, 
        label='Predict (68.2% int)'
    )
    ax1.errorbar(
        xs, np.real(med),
        yerr=np.real(err95.T),
        fmt='none',
        ecolor=color,
        elinewidth=2.0, 
        alpha=0.4, 
        label='Predict (95.4% int)'
    )
    
    # imag component
    ax2.scatter(xs, np.imag(med), color=color, marker='x', label='Predict (median)')
    ax2.errorbar(
        xs, np.imag(med),
        yerr=np.imag(err68.T),
        fmt='none', 
        ecolor=color, 
        elinewidth=2.0, 
        alpha=1.0, 
        label='Predict (68.2% int)'
    )
    ax2.errorbar(
        xs, np.imag(med),
        yerr=np.imag(err95.T),
        fmt='none', 
        ecolor=color, 
        elinewidth=2.0, 
        alpha=0.4, 
        label='Predict (95.4% int)'
    )

def plot_data_over_phi_grid(
    phis_real, phis_imag,
    z,
    title="",
    cmap='RdBu',
    figsize=(16,8),
):
    zr, zi = np.real(z), np.imag(z)
    xv, yv = np.meshgrid(phis_real, phis_imag, indexing='ij')
    
    fig, (axr, axi) = plt.subplots(1, 2, figsize=figsize)
    
    # real component
    zr_min, zr_max = np.min(zr), np.max(zr)
    cr = axr.pcolormesh(xv, yv, zr, cmap=cmap, vmin=zr_min, vmax=zr_max)

    # imag component
    zi_min, zi_max = np.min(zi), np.max(zi)
    ci = axi.pcolormesh(xv, yv, zi, cmap=cmap, vmin=zi_min, vmax=zi_max)

    axr.set_title(f"Re({title})")
    axi.set_title(f"Im({title})")
    fig.colorbar(cr, ax=axr)
    fig.colorbar(ci, ax=axi)
    for ax in (axr, axi):
        ax.axis([xv.min(), xv.max(), yv.min(), yv.max()])
        ax.set_xlabel(r"Re($\phi$)")
        ax.set_ylabel(r"Im($\phi$)")
    plt.tight_layout()
    return fig, (axr, axi)
