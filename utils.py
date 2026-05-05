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

from matplotlib import patches as mpatches


def _skew_window_metrics(
    trajectory,
    data_list_local,
    times_axis,
    window_len_bins,
    top_k_eigs,
    pred_rhs_transpose,
):
    """Per-window |Im lambda|, |Re lambda|, optional bias, and R2; pred is X @ M or X @ M.T."""
    if not trajectory:
        return None
    n_times_loc = len(times_axis)
    centers_ms = []
    omega_mat_im = np.full((len(trajectory), top_k_eigs), np.nan)
    omega_mat_re = np.full((len(trajectory), top_k_eigs), np.nan)
    bias_mat = None
    if len(trajectory[0]) >= 4:
        bias_mat = np.full((len(trajectory), top_k_eigs), np.nan)
    r2_vals = []

    for row_idx, item in enumerate(trajectory):
        if len(item) >= 4:
            center_bin, center_ms, M_win, b_win = item
            b_win = np.asarray(b_win).ravel()
        else:
            center_bin, center_ms, M_win = item
            b_win = None

        centers_ms.append(center_ms)
        w = np.linalg.eigvals(M_win)
        # Preserve pairing between Im and Re while keeping unique |Im| components.
        order_all = np.argsort(np.abs(np.imag(w)))[::-1]
        picked = []
        seen_im = set()
        for idx in order_all:
            im_key = float(np.round(np.abs(np.imag(w[idx])), 12))
            if im_key in seen_im:
                continue
            seen_im.add(im_key)
            picked.append(idx)
            if len(picked) >= top_k_eigs:
                break
        order = np.asarray(picked, dtype=int)
        im_vals = np.round(np.abs(np.imag(w))[order], 12)
        re_vals = np.round(np.real(w)[order], 12)
        omega_mat_im[row_idx, : len(im_vals)] = im_vals
        omega_mat_re[row_idx, : len(re_vals)] = re_vals
        if bias_mat is not None and b_win is not None:
            bias_vals = np.round(b_win, 12)[:top_k_eigs]
            bias_mat[row_idx, : len(bias_vals)] = bias_vals

        bins = bins_for_window(center_bin, window_len_bins, n_times_loc)
        X_w, Xdot_w = stack_XY_in_window(data_list_local, bins)
        pred = X_w @ M_win.T if pred_rhs_transpose else X_w @ M_win
        if b_win is not None:
            pred = pred + b_win

        # R2 = 1 - SSE / SST, where SST uses centered targets.
        sse = np.linalg.norm(Xdot_w - pred, ord="fro") ** 2
        Xdot_centered = Xdot_w - np.mean(Xdot_w, axis=0, keepdims=True)
        sst = np.linalg.norm(Xdot_centered, ord="fro") ** 2
        r2_vals.append(1.0 - sse / sst if sst > 0 else np.nan)

    result = [np.asarray(centers_ms), omega_mat_im, omega_mat_re, np.asarray(r2_vals)]
    if bias_mat is not None:
        result.append(bias_mat)
    return tuple(result)


def diagnose_skew_trajectory(
    trajectory,
    data_list_local,
    times_axis,
    window_len_bins,
    title="",
    top_k_eigs=3,
    restrict_ms=(-50, 150),
    M_skew_global=None,
    M_skew_global_label="global fit (time-invariant)",
    pred_rhs_transpose=False,
    trajectory_secondary=None,
    M_skew_global_secondary=None,
    M_skew_global_secondary_label=None,
    pred_rhs_transpose_secondary=False,
    secondary_series_label="skew_sym_regress",
    primary_series_label="skew_sym_regress",
    plot_real_part=False,
):
    """Spectra + R2 vs window center; optional second trajectory (e.g. skew_sym_regress).

    Parameters
    ----------
    plot_real_part : bool, default False
        If True, create a 3-row subplot with |Im λ|, Re λ, and R^2.
        If False, create a 2-row subplot with |Im λ| and R^2.
    """
    if not trajectory:
        print("empty trajectory:", title)
        return

    pr = _skew_window_metrics(
        trajectory,
        data_list_local,
        times_axis,
        window_len_bins,
        top_k_eigs,
        pred_rhs_transpose,
    )
    if pr is None:
        return

    if len(pr) == 5:
        centers_ms, omega_primary_im, omega_primary_re, r2_primary, bias_primary = pr
        primary_has_bias = True
    else:
        centers_ms, omega_primary_im, omega_primary_re, r2_primary = pr
        bias_primary = None
        primary_has_bias = False

    secondary = None
    secondary_has_bias = False
    if trajectory_secondary is not None:
        secondary = _skew_window_metrics(
            trajectory_secondary,
            data_list_local,
            times_axis,
            window_len_bins,
            top_k_eigs,
            pred_rhs_transpose_secondary,
        )
        if secondary is not None and len(secondary) == 5:
            secondary_has_bias = True

    has_bias_row = primary_has_bias or secondary_has_bias
    n_rows = 2 + int(plot_real_part) + int(has_bias_row)
    figsize_height = 3.5 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, figsize_height), sharex=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    if restrict_ms is not None:
        lo, hi = restrict_ms
        for ax in axes:
            ax.axvspan(lo, hi, alpha=0.18, color="tab:purple", zorder=0)

    row = 0

    # Row 0: |Im λ|
    for k in range(top_k_eigs):
        axes[row].plot(
            centers_ms,
            omega_primary_im[:, k],
            "-o",
            ms=3,
            color=f"C{k}",
            label=f"{primary_series_label} |Im λ| rank {k+1}",
        )

    if secondary is not None:
        secondary_centers, omega_sec_im, omega_sec_re, r2_sec = secondary[:4]
        for k in range(top_k_eigs):
            axes[row].plot(
                secondary_centers,
                omega_sec_im[:, k],
                "--s",
                ms=3,
                color=f"C{k}",
                alpha=0.85,
                label=f"{secondary_series_label} |Im λ| rank {k+1}",
            )

    if M_skew_global is not None:
        wg = np.linalg.eigvals(M_skew_global)
        order_all_g = np.argsort(np.abs(np.imag(wg)))[::-1]
        picked_g = []
        seen_im_g = set()
        for idx in order_all_g:
            im_key = float(np.round(np.abs(np.imag(wg[idx])), 12))
            if im_key in seen_im_g:
                continue
            seen_im_g.add(im_key)
            picked_g.append(idx)
            if len(picked_g) >= top_k_eigs:
                break
        order_g = np.asarray(picked_g, dtype=int)
        im_g = np.round(np.abs(np.imag(wg))[order_g], 12)
        for k in range(top_k_eigs):
            if k >= len(im_g) or np.isnan(im_g[k]):
                continue
            axes[row].axhline(
                im_g[k],
                color=f"C{k}",
                ls=":",
                lw=1.5,
                alpha=0.95,
                label=f"{M_skew_global_label} |Im λ| rank {k+1}",
            )

    if M_skew_global_secondary is not None:
        wg2 = np.linalg.eigvals(M_skew_global_secondary)
        order_all_g2 = np.argsort(np.abs(np.imag(wg2)))[::-1]
        picked_g2 = []
        seen_im_g2 = set()
        for idx in order_all_g2:
            im_key = float(np.round(np.abs(np.imag(wg2[idx])), 12))
            if im_key in seen_im_g2:
                continue
            seen_im_g2.add(im_key)
            picked_g2.append(idx)
            if len(picked_g2) >= top_k_eigs:
                break
        order_g2 = np.asarray(picked_g2, dtype=int)
        im_g2 = np.round(np.abs(np.imag(wg2))[order_g2], 12)
        for k in range(top_k_eigs):
            if k >= len(im_g2) or np.isnan(im_g2[k]):
                continue
            axes[row].axhline(
                im_g2[k],
                color=f"C{k}",
                ls="-.",
                lw=1.5,
                alpha=0.95,
                label=f"{M_skew_global_secondary_label} |Im λ| rank {k+1}",
            )

    axes[row].set_ylabel("|Im λ|")
    axes[row].set_title(title + " — eigenvalue imaginary parts (paired by |Im|)")
    axes[row].axvline(0, color="k", ls="--", lw=0.8, zorder=2)

    leg_handles, leg_labels = axes[row].get_legend_handles_labels()
    if restrict_ms is not None:
        lo, hi = restrict_ms
        leg_handles.append(
            mpatches.Patch(
                facecolor="tab:purple",
                alpha=0.18,
                edgecolor="none",
                label=f"previous preprocess crop ({lo}...{hi} ms)",
            )
        )
    axes[row].legend(handles=leg_handles, fontsize=6, loc="best")
    axes[row].set_xlim(-50, 550)
    row += 1

    # Row 1: Re λ if requested
    if plot_real_part:
        ax_re = axes[row]
        for k in range(top_k_eigs):
            ax_re.plot(
                centers_ms,
                omega_primary_re[:, k],
                "-o",
                ms=3,
                color=f"C{k}",
                label=f"{primary_series_label} Re λ rank {k+1}",
            )

        if secondary is not None:
            for k in range(top_k_eigs):
                ax_re.plot(
                    secondary_centers,
                    omega_sec_re[:, k],
                    "--s",
                    ms=3,
                    color=f"C{k}",
                    alpha=0.85,
                    label=f"{secondary_series_label} Re λ rank {k+1}",
                )

        if M_skew_global is not None:
            wg = np.linalg.eigvals(M_skew_global)
            order_all_g = np.argsort(np.abs(np.imag(wg)))[::-1]
            picked_g = []
            seen_im_g = set()
            for idx in order_all_g:
                im_key = float(np.round(np.abs(np.imag(wg[idx])), 12))
                if im_key in seen_im_g:
                    continue
                seen_im_g.add(im_key)
                picked_g.append(idx)
                if len(picked_g) >= top_k_eigs:
                    break
            order_g = np.asarray(picked_g, dtype=int)
            re_g = np.round(np.real(wg)[order_g], 12)
            for k in range(top_k_eigs):
                if k >= len(re_g) or np.isnan(re_g[k]):
                    continue
                ax_re.axhline(
                    re_g[k],
                    color=f"C{k}",
                    ls=":",
                    lw=1.5,
                    alpha=0.95,
                    label=f"{M_skew_global_label} Re λ rank {k+1}",
                )

        if M_skew_global_secondary is not None:
            wg2 = np.linalg.eigvals(M_skew_global_secondary)
            order_all_g2 = np.argsort(np.abs(np.imag(wg2)))[::-1]
            picked_g2 = []
            seen_im_g2 = set()
            for idx in order_all_g2:
                im_key = float(np.round(np.abs(np.imag(wg2[idx])), 12))
                if im_key in seen_im_g2:
                    continue
                seen_im_g2.add(im_key)
                picked_g2.append(idx)
                if len(picked_g2) >= top_k_eigs:
                    break
            order_g2 = np.asarray(picked_g2, dtype=int)
            re_g2 = np.round(np.real(wg2)[order_g2], 12)
            for k in range(top_k_eigs):
                if k >= len(re_g2) or np.isnan(re_g2[k]):
                    continue
                ax_re.axhline(
                    re_g2[k],
                    color=f"C{k}",
                    ls="-.",
                    lw=1.5,
                    alpha=0.95,
                    label=f"{M_skew_global_secondary_label} Re λ rank {k+1}",
                )

        ax_re.set_ylabel("Re λ")
        ax_re.set_title(title + " — eigenvalue real parts (paired by |Im|)")
        ax_re.axvline(0, color="k", ls="--", lw=0.8, zorder=2)
        ax_re.legend(fontsize=6, loc="best")
        ax_re.set_xlim(-50, 550)
        row += 1

    # Row 2: bias term if present
    if has_bias_row:
        ax_b = axes[row]
        if bias_primary is not None:
            for k in range(top_k_eigs):
                ax_b.plot(
                    centers_ms,
                    bias_primary[:, k],
                    "-o",
                    ms=3,
                    color=f"C{k}",
                    label=f"{primary_series_label} bias b[{k+1}]",
                )
        if secondary is not None and len(secondary) == 5:
            _, _, _, _, bias_sec = secondary
            for k in range(top_k_eigs):
                ax_b.plot(
                    secondary_centers,
                    bias_sec[:, k],
                    "--s",
                    ms=3,
                    color=f"C{k}",
                    alpha=0.85,
                    label=f"{secondary_series_label} bias b[{k+1}]",
                )
        ax_b.set_ylabel("bias b")
        ax_b.set_title(title + " — affine bias term")
        ax_b.axvline(0, color="k", ls="--", lw=0.8, zorder=2)
        ax_b.legend(fontsize=6, loc="best")
        ax_b.set_xlim(-50, 550)
        row += 1

    # Last row: R^2
    ax_r2 = axes[row]
    ax_r2.plot(
        centers_ms,
        r2_primary,
        color="C3",
        marker="o",
        ms=3,
        ls="-",
        label=f"per-window {primary_series_label} ($R^2$)",
    )

    if secondary is not None:
        _, _, _, r2_sec = secondary[:4]
        ax_r2.plot(
            secondary_centers,
            r2_sec,
            color="C2",
            marker="s",
            ms=3,
            ls="--",
            label=f"per-window {secondary_series_label} ($R^2$)",
        )

        # Global R2 on the same windows (fair comparison).
        r2_globs_pri = []
        r2_globs_sec = []
        for i, item in enumerate(trajectory):
            center_bin = item[0]
            bins = bins_for_window(center_bin, window_len_bins, len(times_axis))
            X_w, Xdot_w = stack_XY_in_window(data_list_local, bins)
            Xdot_centered = Xdot_w - np.mean(Xdot_w, axis=0, keepdims=True)
            sst = np.linalg.norm(Xdot_centered, ord="fro") ** 2
            if sst <= 0:
                r2_globs_pri.append(np.nan)
                r2_globs_sec.append(np.nan)
                continue
            if M_skew_global is not None:
                pg = X_w @ M_skew_global.T if pred_rhs_transpose else X_w @ M_skew_global
                sse_g = np.linalg.norm(Xdot_w - pg, ord="fro") ** 2
                r2_globs_pri.append(1.0 - sse_g / sst)
            if M_skew_global_secondary is not None:
                ps = (
                    X_w @ M_skew_global_secondary.T
                    if pred_rhs_transpose_secondary
                    else X_w @ M_skew_global_secondary
                )
                sse_s = np.linalg.norm(Xdot_w - ps, ord="fro") ** 2
                r2_globs_sec.append(1.0 - sse_s / sst)

        if M_skew_global is not None and r2_globs_pri:
            ax_r2.plot(
                centers_ms,
                r2_globs_pri,
                color="C0",
                ls=":",
                lw=1.5,
                marker="x",
                ms=4,
                label=f"{M_skew_global_label} ($R^2$)",
            )
        if M_skew_global_secondary is not None and any(np.isfinite(r2_globs_sec)):
            ax_r2.plot(
                centers_ms,
                r2_globs_sec,
                color="C1",
                ls="-.",
                lw=1.5,
                marker="+",
                ms=5,
                label=f"{M_skew_global_secondary_label} ($R^2$)",
            )
    else:
        # Original single-series global overlay, now in R2 form.
        if M_skew_global is not None:
            r2_global = []
            for item in trajectory:
                center_bin = item[0]
                bins = bins_for_window(center_bin, window_len_bins, len(times_axis))
                X_w, Xdot_w = stack_XY_in_window(data_list_local, bins)
                pred_g = X_w @ M_skew_global.T if pred_rhs_transpose else X_w @ M_skew_global
                sse_g = np.linalg.norm(Xdot_w - pred_g, ord="fro") ** 2
                Xdot_centered = Xdot_w - np.mean(Xdot_w, axis=0, keepdims=True)
                sst = np.linalg.norm(Xdot_centered, ord="fro") ** 2
                r2_global.append(1.0 - sse_g / sst if sst > 0 else np.nan)
            ax_r2.plot(
                centers_ms,
                r2_global,
                color="C0",
                marker="s",
                ms=3,
                label=f"{M_skew_global_label} ($R^2$)",
            )

    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.set_title(r"window-wise $R^2$ (higher is better; multiply convention matches each fit)")
    ax_r2.axvline(0, color="k", ls="--", lw=0.8, zorder=2)
    ax_r2.axhline(0, color="k", ls="--", lw=0.8, zorder=2)
    ax_r2.set_ylim(-0.2, 1.00)
    ax_r2.set_xlabel("window center (ms)")
    ax_r2.legend(fontsize=7, loc="best")
    ax_r2.set_xlim(-50, 550)

    plt.tight_layout()
    plt.show()

def fit_affine_ridge(X, Xdot, lam=1e-2, penalize_bias=False):
    """
    Fit Xdot ≈ X M + b using ridge regression.
    Row-vector convention.
    """
    N, D = X.shape

    X_aug = np.hstack([X, np.ones((N, 1))])

    I = np.eye(D + 1)
    if not penalize_bias:
        I[-1, -1] = 0.0  # do not penalize b

    B = np.linalg.solve(
        X_aug.T @ X_aug + lam * I,
        X_aug.T @ Xdot
    )

    M = B[:-1, :]
    b = B[-1, :]

    return M, b

def fixed_point_damped(M, b, eps=1e-2):
    """
    Solve x M + b = 0 with Tikhonov damping.
    Equivalent to x = -b M^T (M M^T + eps I)^(-1)
    """
    D = M.shape[0]
    return -b @ M.T @ np.linalg.inv(M @ M.T + eps * np.eye(D))