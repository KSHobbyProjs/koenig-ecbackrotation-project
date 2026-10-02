"""
utils.py
"""

import numpy as np
from pathlib import Path

from typing import Callable, Sequence
from dataclasses import dataclass

from .sweep_utils import ScanPoint
from ..dvr import DVR
from ..ec import ECSystem
from ..ecensemble import StatPoint

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker

# ===============================================================================================================
#                                                      PRINTING
# ===============================================================================================================
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
    
# ===============================================================================================================
#                                                      PLOTTING
# ===============================================================================================================
@dataclass
class PlotObject:
    fig: matplotlib.figure.Figure
    axs: matplotlib.axes.Axes | Sequence[matplotlib.axes.Axes]

    def save(self, plots_folder, filename="", filetype="png", **kwargs):
        self.fig.savefig(plots_folder / f"{filename}.{filetype}", **kwargs)

    def show(self):
        """ Show the figure. This is not needed in Jupyter as Jupyter naturally plots the figures """
        self.fig.show()

    def close(self):
        plt.close(self.fig)

def get_plots_path():
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "plots"
    
def plot_observables(
    xs: np.ndarray,
    observables: dict[str, np.ndarray],
    *,
    xlabel: str="",
    ylabels: dict[str, str] = {},
    titles: dict[str, str] = {},
    figsizes: dict[str, tuple[int, int]] | tuple[int, int]=(12, 6)
) -> dict[str, PlotObject]:
    
    figsizes = figsizes if isinstance(figsizes, dict) else {name: figsizes for name in observables.keys()}

    plots = {}
    for name, vals in observables.items():
        # get the ylabel and title if given, otherwise default to name of observable
        ylabel = ylabels.get(name, name)
        title = titles.get(name, name)
        figsize = figsizes.get(name)
        
        fig, (axr, axi) = plt.subplots(1, 2, figsize=figsize)
        # real plot
        axr.plot(xs, np.real(vals))
        axr.set_ylabel(fr"$\Re$ {ylabel}")
        axr.set_xlabel(xlabel)
        # imag plot
        axi.plot(xs, np.imag(vals))
        axi.set_ylabel(fr"$\Im$ {ylabel}")
        axi.set_xlabel(xlabel)

        fig.suptitle(title)
        fig.tight_layout()

        plots[name] = PlotObject(fig=fig, axs=(axr, axi))
    return plots

def plot_observables_over_phi_grid(
    phis_real: np.ndarray,
    phis_imag: np.ndarray,
    observables: dict[str, np.ndarray],
    *,
    titles: dict[str, str] = {},
    cmap: str='RdBu',
    figsizes: dict[str, tuple[int, int]] | tuple[int, int]=(16, 8),
) -> dict[str, PlotObject]:

    figsizes = figsizes if isinstance(figsizes, dict) else {name: figsizes for name in observables.keys()}

    plots = {}
    for name, vals in observables.items():
        title = titles.get(name, name)
        figsize = figsizes.get(name)

        zr, zi = np.real(vals), np.imag(vals)
        xv, yv = np.meshgrid(phis_real, phis_imag, indexing='ij')
    
        fig, (axr, axi) = plt.subplots(1, 2, figsize=figsize)
    
        # real component
        zr_min, zr_max = np.min(zr), np.max(zr)
        cr = axr.pcolormesh(xv, yv, zr, cmap=cmap, vmin=zr_min, vmax=zr_max)
        fig.colorbar(cr, ax=axr)
        axr.set_title(fr"$\Re$ {title}")

        # imag component
        zi_min, zi_max = np.min(zi), np.max(zi)
        ci = axi.pcolormesh(xv, yv, zi, cmap=cmap, vmin=zi_min, vmax=zi_max)
        fig.colorbar(ci, ax=axi)
        axi.set_title(fr"$\Im$ {title}")
        
        for ax in (axr, axi):
            ax.axis([xv.min(), xv.max(), yv.min(), yv.max()])
            ax.set_xlabel(r"$\Re\phi$")
            ax.set_ylabel(r"$\Im\phi$")
        
        fig.tight_layout()

        plots[name] = PlotObject(fig=fig, axs=(axr, axi))
    return plots

def plot_ensemble_stats(
    xs: np.ndarray,
    ensemble_stats: list[dict[str, StatPoint]],
    references: dict[str, complex]={},
    *,
    xlabel: str="",
    ylabels: dict[str, str] = {},
    titles: dict[str, str] = {},
    color: str='red',
    figsizes: dict[str, tuple[int, int]] | tuple[int, int]=(12,6),
    scale: str='linear',
    cutoff: int=0
) -> dict[str, PlotObject]:
    """ 
    Plots EC ensemble stats with error bands. Designed to plot data and error bars as number of training
    points varies, but `xs` can be anything.

    Parameters
    ----------
    xs: np.ndarray
        The x data over which the stats were computed. Shape (len(xs),).
    ensemble_results: list[dict[str, StatPoint]]
        List of ensemble stats at each x in `xs`.
    references: dict[str, complex] (optional)
        Reference data for every observable. When given, adds a reference line to the plot.
    cutoff: int
        Cuts off the first "cutoff" data points. For example, if cutoff=1, then only xs[1:], med[1:], etc.
        will be plotted. Default is no cutoff.
    """
    xs = np.asarray(xs)[cutoff:]
    figsizes = figsizes if isinstance(figsizes, dict) else {name: figsizes for name in ensemble_stats[0].keys()}

    # re-structure so that list[dict[str, StatPoint]] -> dict[str, list[list[complex], list[complex], list[complex]]]
    # such that ensemble_stats[i]["observable_j"] -> result["observable_j"] = [med[i], err68[i], err95[i]]
    stats = {}
    for name in ensemble_stats[0]:
        med, err68, err95 = [], [], []
        for stat in ensemble_stats[cutoff:]:
            med.append(stat[name].med)
            err68.append(stat[name].err68)
            err95.append(stat[name].err95)
        stats[name] = (np.array(med), np.array(err68), np.array(err95))

    plots = {}
    for name, vals in stats.items():
        title = titles.get(name, name)
        ylabel = ylabels.get(name, name)
        exact = references.get(name)
        figsize = figsizes.get(name)
        
        med, err68, err95 = vals

        fig, (axr, axi) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)
        # real component 
        axr.scatter(xs, np.real(med), color=color, marker='x', label='Predict (median)')
        axr.errorbar(xs, np.real(med), yerr=np.real(err68.T), fmt='none', ecolor=color, 
            elinewidth=2.0, alpha=1.0, label='Predict (68.2% int)')
        axr.errorbar(xs, np.real(med), yerr=np.real(err95.T), fmt='none', ecolor=color,
            elinewidth=2.0, alpha=0.4, label='Predict (95.4% int)')
        axr.set_ylabel(rf"$\Re$ {ylabel}")
        
        # imag component
        axi.scatter(xs, np.imag(med), color=color, marker='x', label='Predict (median)')
        axi.errorbar(xs, np.imag(med), yerr=np.imag(err68.T), fmt='none', ecolor=color, 
            elinewidth=2.0, alpha=1.0, label='Predict (68.2% int)')
        axi.errorbar(xs, np.imag(med), yerr=np.imag(err95.T), fmt='none', ecolor=color, 
            elinewidth=2.0, alpha=0.4, label='Predict (95.4% int)')
        axi.set_xlabel(xlabel)
        axi.set_ylabel(rf"$\Im$ {ylabel}")

        # if given a reference, plot it as a line
        if exact is not None:
            axr.axhline(y=np.real(exact), color='black', linewidth=1.0, linestyle='-', label='Exact')
            axi.axhline(y=np.imag(exact), color='black', linewidth=1.0, linestyle='-', label='Exact')

        for ax in (axr, axi):
            ax.set_xlabel(xlabel)
            ax.set_yscale(scale)

        # set legend
        handles, labels = axr.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.5)
        fig.tight_layout()

        plots[name] = PlotObject(fig=fig, axs=(axr, axi))
    return plots