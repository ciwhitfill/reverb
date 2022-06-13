"""
Matplotlib theme
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, EngFormatter)
import seaborn as sns
import numpy as np
from scipy import signal

color_list = sns.color_palette("husl", 16)

sns.set_style("white")


sns.set(rc={
    'figure.dpi': 300,
    'figure.figsize': [5.333, 3],
    'font.serif': 'Museo',
    'font.sans-serif': 'Museo Sans',
    'font.family': 'serif',
    'text.usetex': False,
    'font.weight': 300,
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',
    'axes.grid': False,
    'grid.alpha': 0.2,
    'grid.color': 'lightgrey',
    'axes.labelcolor': 'dimgrey',
    'axes.labelweight': 400,
    'axes.titleweight': 600,
    'axes.spines.top': False,
    'axes.prop_cycle': plt.cycler(color=color_list),
    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.bottom': True,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'in',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'in',
    'ytick.left': True,
    'ytick.right': False})

sns.set_context("notebook", rc={'font.size': 8,
                                'axes.titlesize': 16,
                                'axes.labelsize': 9,
                                'xtick.labelsize': 8,
                                'ytick.labelsize': 8,
                                'lines.linewidth': 1,
                                'axes.linewidth': 1,
                                'xtick.major.width': 1,
                                'xtick.minor.width': 0.5,
                                'ytick.major.width': 1,
                                'ytick.minor.width': 0.5})


def plot_ir(x: np.array, ir: np.array, color: str):
    """Plot amplitude over time"""
    fig, ax = plt.subplots()
    ax.stem(x, ir, linefmt=color, markerfmt=' ', basefmt=' ')
    ax.set_title('IMPULSE RESPONSE', fontdict={'family': 'sans-serif'})
    ax.axis('tight')
    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))

    sns.despine(right=True)

    return fig, ax

def plot_fr(ir: np.array, sample_rate: int, color1: str, color2: str, y1_lim=None, y2_lim=None, y1_inc=3, y2_inc=30):
    """Plot amplitude and phase over frequency"""
    w, h = signal.freqz(ir, fs=sample_rate)
    fig, ax1 = plt.subplots()
    
    ax1.set_title('FREQUENCY RESPONSE', fontdict={'family': 'sans-serif'})
    ax1.semilogx(w, 20 * np.log10(np.abs(h)), color1)
    if y1_lim:
        ax1.set_ylim(y1_lim)

    ax1.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax1.yaxis.set_major_locator(MultipleLocator(y1_inc))
    ax1.yaxis.set_major_formatter(EngFormatter(unit='dB'))
    ax1.tick_params(axis='x', which='minor', colors='lightgrey')
    
    ax2 = ax1.twinx()
    if y2_lim:
        ax2.set_ylim(y2_lim)
    angles = np.unwrap(np.angle(h, deg=True))
    ax2.plot(w, angles, c=color2, ls='--')
    ax2.yaxis.set_major_locator(MultipleLocator(y2_inc))
    ax2.yaxis.set_major_formatter(EngFormatter(unit='°', sep=''))

    ax2.yaxis.tick_right()
    
    return fig, ax1, ax2