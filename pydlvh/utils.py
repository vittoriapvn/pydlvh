"""
utils.py
========
Internal helper functions for pyDLVH.

These are not part of the public API and may change without notice.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from typing import Optional, Tuple, Iterable, Literal


def _freedman_diaconis_bins(*, data: np.ndarray, max_bins: int = 200) -> np.ndarray:
    """
    Suggest bin edges using the Freedman–Diaconis rule.

    Parameters
    ----------
    data : np.ndarray
        Input 1D or ND array (flattened internally).
    max_bins : int, default=200
        Maximum number of bins allowed.

    Returns
    -------
    bins : np.ndarray
        Array of bin edges covering the data range.

    Notes
    -----
    h = 2 * IQR(x) / n^(1/3). If h <= 0 or range == 0, fallback.
    """
    x = np.asarray(data).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        raise ValueError("Empty data passed to _freedman_diaconis_bins.")
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # Handle constant array (range == 0) → make a tiny 2-edge range
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Non-finite range in _freedman_diaconis_bins.")
    if xmax <= xmin:
        eps = 1e-6 if xmin == 0.0 else abs(xmin) * 1e-6
        return np.array([xmin, xmin + eps], dtype=float)

    if n < 2:
        return np.array([xmin, xmax], dtype=float)

    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    h = 2.0 * iqr / (n ** (1.0 / 3.0)) if iqr > 0 else 0.0

    if h <= 0:
        std = float(np.std(x))
        h = (2.0 * std) / (n ** (1.0 / 3.0)) if std > 0 else (xmax - xmin)

    nbins = int(np.ceil((xmax - xmin) / h)) if h > 0 else 1
    nbins = min(max(nbins, 1), max_bins)

    return np.linspace(xmin, xmax, nbins + 1)


def _auto_bins(*, array: np.ndarray, max_bins: int = 200) -> np.ndarray:
    """
    Suggest optimal bin edges for a 1D histogram.

    Returns
    -------
    edges : np.ndarray
        Bin edges covering the data range.
    """
    return _freedman_diaconis_bins(data=array, max_bins=max_bins)


def _suffix_cumsum2d(counts: np.ndarray) -> np.ndarray:
    """Fast suffix cumulative sum for 2D arrays.
    Given differential counts C[i,j] on an (Nd×Nl) grid, returns S where
    S[i,j] = sum_{p>=i, q>=j} C[p,q].
    """
    # reverse both axes → prefix cumsum → reverse back
    s = counts[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    return s


def suggest_common_edges(
    *,
    arrays: Iterable[np.ndarray],
    method: Literal["fd", "range"] = "fd",
    max_bins: int = 200,
    bin_width: Optional[float] = None,
) -> np.ndarray:
    """
    Suggest a common set of bin edges for multiple 1D arrays.

    Parameters
    ----------
    arrays : iterable of np.ndarray
        Collections of arrays to pool (flattened internally).
    method : {"fd", "range"}
        - "fd": Freedman–Diaconis on pooled data (default).
        - "range": if bin_width is provided, builds edges on [min,max] of pooled data.
    max_bins : int
        Cap for FD bins.
    bin_width : float, optional
        If provided (and method="range"), use fixed bin width across the pooled range.

    Returns
    -------
    edges : np.ndarray
        Common bin edges.
    """
    pooled = np.concatenate([np.asarray(a).ravel() for a in arrays])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError("No finite data in suggest_common_edges().")

    if method == "fd" and bin_width is None:
        return _freedman_diaconis_bins(data=pooled, max_bins=max_bins)

    # method == "range" or custom width
    xmin = float(np.min(pooled))
    xmax = float(np.max(pooled))
    if bin_width is None:
        # fallback to FD if no width
        return _freedman_diaconis_bins(data=pooled, max_bins=max_bins)

    nb = int(np.ceil((xmax - xmin) / bin_width)) if bin_width > 0 else 1
    nb = max(nb, 1)
    return np.linspace(0.0, nb * bin_width, nb + 1)


def suggest_common_edges_2d(
    *,
    dose_arrays: Iterable[np.ndarray],
    let_arrays: Iterable[np.ndarray],
    dose_method: Literal["fd", "range"] = "fd",
    let_method: Literal["fd", "range"] = "fd",
    max_bins_dose: int = 200,
    max_bins_let: int = 200,
    dose_bin_width: Optional[float] = None,
    let_bin_width: Optional[float] = None,
    ) -> np.ndarray:
    """Suggest common (dose_edges, let_edges) for a cohort."""
    d_edges = suggest_common_edges(
        arrays=dose_arrays, method=dose_method, max_bins=max_bins_dose, bin_width=dose_bin_width
    )
    l_edges = suggest_common_edges(
        arrays=let_arrays, method=let_method, max_bins=max_bins_let, bin_width=let_bin_width
    )
    return d_edges, l_edges


def _get_bin_edges(*, centers: np.ndarray,
                   first_edge: Optional[float] = None,
                   last_edge: Optional[float] = None) -> np.ndarray:

    centers = np.asarray(centers, dtype=float)
    # Sort centers in ascending order
    centers = np.sort(centers)

    # Compute bin edges from centers
    edges = (centers[:-1] + centers[1:]) / 2

    # Determine the first edge (based on the width of the first bin)
    if first_edge is None:
        first_bin_width = (centers[1] - centers[0]) / 2
        first_edge = centers[0] - first_bin_width
        if first_edge < 0:
            first_edge = 0.0
    else:
        if first_edge > centers[0]:
            raise ValueError("first_edge must be less than the first center value.")

    # Determine the last edge (based on the width of the last bin)
    if last_edge is None:
        if len(centers) < 2:
            last_edge = centers[-1]
        else:
            last_bin_width = (centers[-1] - centers[-2]) / 2
            last_edge = centers[-1] + last_bin_width
    else:
        if last_edge <= centers[-1]:
            raise ValueError("last_edge must be greater than the last center value.")
        
    edges = np.concatenate(([first_edge], edges, [last_edge]))

    return edges


def _get_bin_centers(*, edges: np.ndarray) -> np.ndarray:

    edges = np.asarray(edges, dtype=float)
    # Sort centers in ascending order
    edges = np.sort(edges)

    # Compute bin centers from edges
    centers = (edges[:-1] + edges[1:]) / 2

    return centers


def project_contours_to_DVH(fig, axr, axb, CS, isovolumes_colors, ls, DVH_interp, y_max=120):
    for path, color in zip(CS.get_paths(), isovolumes_colors):
        dose = path.vertices[:, 0]
        d = dose.max()
        y_stop = float(DVH_interp(d))

        axb.vlines(
            d,
            ymin=y_stop,
            ymax=y_max,
            color=color,
            lw=1.5,
            linestyle=ls,
            alpha=1,
            zorder=5,
        )

        axb.plot(d, y_stop, "x", color=color, markersize=5)

        _connect_axes(fig,
                      axr,  d, 0,     # DLVH point
                      axb, d, 110,   # DVH intersection
                      color=color, lw=1.5, ls=ls)

def project_contours_to_LVH(fig, axr, axl, CS, isovolumes_colors, ls, LVH_interp, x_min=-100):

    for path, color in zip(CS.get_paths(), isovolumes_colors):
        let = path.vertices[:, 1]
        l=let.max()
        x_stop = float(LVH_interp(l))

        axl.hlines(
            l,
            xmin=x_min,
            xmax=x_stop,
            color=color,
            linestyle=ls,
            lw=1.5,
            alpha=1,
            zorder=5,
        )

        axl.plot(x_stop, l, "x", color=color, markersize=5)
        
        _connect_axes(fig,
                      axr,  0, l,   # DLVH point
                      axl, -5, l,   # LVH intersection
                      color=color, lw=1.5, ls=ls)

def _data_to_fig(fig, ax, x, y):
    display = ax.transData.transform((x, y))
    return fig.transFigure.inverted().transform(display)

def _connect_axes(
    fig,
    ax_from, x_from, y_from,
    ax_to,   x_to,   y_to,
    color="black",
    lw=1,
    ls="solid",
):
    x0, y0 = _data_to_fig(fig, ax_from, x_from, y_from)
    x1, y1 = _data_to_fig(fig, ax_to,   x_to,   y_to)

    line = Line2D(
        [x0, x1],
        [y0, y1],
        transform=fig.transFigure,
        color=color,
        lw=lw,
        ls=ls,
        alpha=1,
        zorder=10,
    )
    fig.add_artist(line)

