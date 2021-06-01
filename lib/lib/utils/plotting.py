import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


def plotmat_sidebyside(mats, grid, figsize=(5.5, 1.95), ticks=None, ytitle=None):
    """Plot matrices side-by-side with the same color scheme"""
    labels = list(mats.keys())
    mats = list(mats.values())
    n = len(mats)
    # Build colormap
    vmin = min(map(lambda A: A.min(), mats))
    vmax = max(map(lambda A: A.max(), mats))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'plasma'
    # Plot matrices
    fig, axs = plt.subplots(*grid, figsize=figsize)
    axs = np.ravel(axs)
    for ax, M, label in zip(axs, mats, labels):
        plt.sca(ax)
        ax.invert_yaxis()
        ax.set_aspect(1.0)
        dim = len(M)
        X = np.tile(np.arange(dim+1)+0.5, (dim+1,1))
        Y = X.T
        p = plt.pcolormesh(X, Y, M, norm=norm, cmap=cmap)
        if ticks:
            plt.xticks(ticks)
            plt.yticks(ticks)
        p.cmap.set_over('white')
        p.cmap.set_under('black')
        plt.title(label, pad=10, y=ytitle)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(p, cax=cax)

    return fig
