"""
utils.py
"""

import numpy as np
from pathlib import Path

def decompose_stats(data):
    med = [r[0] for r in data]
    err68 = [r[1] for r in data]
    err95 = [r[2] for r in data]
    return np.array(med), np.array(err68), np.array(err95)

def plot_stats(fig, axs, xs, med, err68, err95, cutoff: int=0, color='red'):
    ax1, ax2 = axs
    xs, med, err68, err95 = xs[cutoff:], med[cutoff:], err68[cutoff:, :], err95[cutoff:, :]
    
    # real component 
    ax1.scatter(
        xs, np.real(med),
        color=color, 
        marker='x', 
        label='Predict (median)'
    )
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
    ax2.scatter(
        xs, np.imag(med),
        color=color,
        marker='x', 
        label='Predict (median)'
    )
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

def get_plots_path():
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "plots"
