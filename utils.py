import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.io import loadmat

from sklearn.decomposition import PCA

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe


def plot_3d_trajectory(ax, x, y, z, 
                    color="black",
                    outline="black",
                    circle=True,
                    arrow=True,
                    circle_size=0.05,
                    arrow_size=0.05):
    """
    Plot a single neural trajectory in a 3D space.
    
    Args
    ----
        ax: Axis used for plotting.
        
        x: Values of variable on x-axis.
        
        y: Values of variable on y-axis.
        
        color: Fill color of line to be plotted. Defaults to "black".
        
        outline: Outline color of line. Defaults to "black".
        
        circle: True if the trajectory should have a circle at its starting state.
        
        arrow: True if the trajectory should have an arrow at its ending state.
        
    """
    # Plot the 3D trajectory line
    ax.plot(x, y, z, color=color, linewidth=2)
        
    if circle:
        # Plot a scatter point at the starting position
        ax.scatter([x[0]], [y[0]], [z[0]], 
                  color=color, s=circle_size*100, 
                  edgecolors="black", linewidth=1)

    if arrow:
        # Use quiver for 3D arrow
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        dz = z[-1] - z[-2]
        px, py, pz = (x[-1], y[-1], z[-1])
        ax.quiver(px, py, pz, dx, dy, dz,
                  color=color,
                  arrow_length_ratio=0.3,
                  linewidth=2)

def plot_3d_projections(data_list,
                     x_idx=0,
                     y_idx=1,
                     z_idx=2,
                     axis=None,
                     arrows=True,
                     circles=True,
                     arrow_size=0.05,
                     circle_size=0.05):
    """
    Plot trajectories found via jPCA or PCA. 
    
    Args
    ----
        data_list: List of trajectories, where each entry of data_list is an array of size T x D, 
                   where T is the number of time-steps and D is the dimension of the projection.
        x_idx: column of data which will be plotted on x axis. Default 0.
        y_idx: column of data which will be plotted on y axis. Default 1.
        z_idx: column of data which will be plotted on z axis. Default 2.
        arrows: True to add arrows to the trajectory plot.
        circles: True to add circles at the beginning of each trajectory.
        sort_colors: True to color trajectories based on the starting x coordinate. This mimics
                     the jPCA matlab toolbox.
    """
    if axis is None:
        fig = plt.figure(figsize=(5,5))
        axis = fig.add_axes([1, 1, 1, 1])

    colormap = plt.cm.RdBu
    colors = np.array([colormap(i) for i in np.linspace(0, 1, len(data_list))])
    data_list = [data[:, [x_idx, y_idx, z_idx]] for data in data_list]

    start_x_list = [data[0,0] for data in data_list]
    color_indices = np.argsort(start_x_list)

    for i, data in enumerate(np.array(data_list)[color_indices]):
        plot_3d_trajectory(axis,
                           data[:, 0],
                           data[:, 1],
                           data[:, 2],
                           color=colors[i],
                           circle=circles,
                        arrow=arrows,
                        arrow_size=arrow_size,
                        circle_size=circle_size)


def bins_for_window(center_bin, window_len_bins, n_times):
    """which time-bin indices fall inside this window (never outside 0 … T-1)."""
    half = window_len_bins // 2
    lo = max(0, center_bin - half)
    hi_excl = min(n_times, center_bin + half + 1)
    return np.arange(lo, hi_excl)


def stack_XY_in_window(data_list, bins):
    bins = np.asarray(bins, dtype=int)
    X_parts, Xdot_parts = [], []
    for x in data_list:
        T = x.shape[0]
        ok = (bins >= 0) & (bins < T - 1)
        r = bins[ok]
        if r.size == 0:
            continue
        X_parts.append(x[r])
        Xdot_parts.append(x[r + 1] - x[r])
    return np.concatenate(X_parts, axis=0), np.concatenate(Xdot_parts, axis=0)