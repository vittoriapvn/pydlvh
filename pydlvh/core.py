from __future__ import annotations

import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Literal, Optional, Union
from .utils import _auto_bins, _suffix_cumsum2d, _get_bin_edges, _get_bin_centers


class Histogram1D:
    """Internal 1D histogram container with plotting capability."""

    def __init__(self, *, values: np.ndarray, edges: np.ndarray,
                 quantity: str, normalize: bool, cumulative: bool,
                 x_label: Optional[str] = None, y_unit: Optional[str] = None,
                 centers: Optional[np.ndarray] = None,
                 err: Optional[np.ndarray] = None,
                 p_lo: Optional[np.ndarray] = None,
                 p_hi: Optional[np.ndarray] = None,
                 stat: Optional[str] = None,
                 aggregatedby: Optional[str] = None):
        self.values = np.asarray(values, dtype=float)
        self.edges = np.asarray(edges, dtype=float)
        self.centers = np.asarray(centers, dtype=float)
        self.quantity = str(quantity)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)
        self.x_label = x_label
        self.y_unit = y_unit

        # Optional cohort statistics
        self.err = None if err is None else np.asarray(err, dtype=float)
        self.p_lo = None if p_lo is None else np.asarray(p_lo, dtype=float)
        self.p_hi = None if p_hi is None else np.asarray(p_hi, dtype=float)
        self.stat = stat if stat else None
        self.aggregated = False if stat else True # Aggregation is deduced from stat declaration
        self.aggregatedby = aggregatedby

    def _get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (edges, values) suitable for ax.step(where="post").

        Convention:
        - step expects len(values) == len(edges)
        - cumulative histograms are typically stored already with a trailing 0 bin
        - differential histograms are stored with len(values) == len(edges)-1 (np.histogram)
          -> we append a trailing 0 for step visualization.
        """
        edges = self.edges.copy()
        values = self.values.copy()

        # Optional visual padding: ensure x starts at 0 for cumulative curves
        if self.cumulative and edges.size > 0 and edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            # keep same length convention by duplicating first value
            values = np.insert(values, 0, values[0])

        # Ensure step-compatible shapes
        if values.size == edges.size - 1:
            # common differential case
            values = np.append(values, 0.0)
        elif values.size != edges.size:
            warnings.warn(
                f"Histogram1D shape mismatch for step plotting: "
                f"len(edges)={edges.size}, len(values)={values.size}. "
                f"Proceeding without shape correction.",
                RuntimeWarning,
            )

        return edges, values
    
    def _get_error(self, *, error: Optional[np.ndarray], values: np.ndarray) -> Optional[np.ndarray]:
        """
        Return error array aligned to 'values' for step plotting.
        Accepts:
        - same shape as values
        - one element shorter (missing trailing 0) -> append 0
        Otherwise returns None with a warning.
        """
        if error is None:
            return None

        err = np.asarray(error, dtype=float)

        if err.shape == values.shape:
            return err

        if err.size == values.size - 1:
            return np.append(err, 0.0)

        warnings.warn(
            f"Histogram1D error band shape mismatch: "
            f"len(values)={values.size}, len(error)={err.size}. Skipping band.",
            RuntimeWarning,
        )
        return None

    def plot(self, *, ax: Optional[plt.Axes] = None,
             show_band: bool = True, band_color: str = None, **kwargs):

        if ax is None:
            _, ax = plt.subplots()
    
        # Plot Histogram1D (accounting for eventual padding)
        edges, values = self._get_data()

        ax.step(edges, values, where="post", **kwargs)
        x_band = edges
        step_kw = "post"
    
        # Plot uncertainty range
        if show_band:
            if not band_color:
                band_color = "gray"

            # std band
            if self.err is not None:
                err = self._get_error(error=self.err, values=values)
                if self.aggregatedby == "volume":
                    x_lo = edges - err
                    x_hi = edges + err
                    ax.fill_betweenx(values, x_lo, x_hi,
                                     alpha=0.2, color=band_color, step=step_kw)
                else:
                    if err is not None:
                        y_lo = values - err
                        y_hi = values + err
                        ax.fill_between(x_band, y_lo, y_hi,
                                        step=step_kw, alpha=0.2, color=band_color)
    
            # percentile band
            if self.p_lo is not None and self.p_hi is not None:
                plo = self._get_error(error=self.p_lo, values=values)
                phi = self._get_error(error=self.p_hi, values=values)
                if self.aggregatedby == "volume":
                    ax.fill_betweenx(values, plo, phi,
                                     alpha=0.2, color=band_color, step=step_kw)
                else:
                    ax.fill_between(x_band, plo, phi,
                                    step=step_kw, alpha=0.2, color=band_color)
    
        # Labels
        if self.x_label:
            ax.set_xlabel(self.x_label)
        elif self.quantity == "dose":
            ax.set_xlabel("Dose [Gy(RBE)]")
        elif self.quantity == "let":
            ax.set_xlabel(r"LET$_{d}$ [keV/µm]")
    
        if self.normalize:
            # ax.set_ylabel(f"{'Cumulative' if self.cumulative else 'Differential'} Volume [%]")
            ax.set_ylabel("Volume [%]")
        else:
            ax.set_ylabel("Volume [cm³]")
    
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return ax


class Histogram2D:
    """Internal 2D histogram container with plotting capability."""

    def __init__(self, *, values: np.ndarray,
                 dose_edges: np.ndarray, let_edges: np.ndarray,
                 normalize: bool, cumulative: bool,
                 dose_label: str = "Dose [Gy]",
                 let_label: str = "LET [keV/µm]",
                 err: Optional[np.ndarray] = None,
                 p_lo: Optional[np.ndarray] = None,
                 p_hi: Optional[np.ndarray] = None,
                 stat: Optional[str] = None):
        
        self.values = np.asarray(values, dtype=float)
        self.dose_edges = np.asarray(dose_edges, dtype=float)
        self.let_edges = np.asarray(let_edges, dtype=float)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)
        self.dose_label = dose_label
        self.let_label = let_label

        # Optional cohort statistics
        self.err = None if err is None else np.asarray(err, dtype=float)
        self.p_lo = None if p_lo is None else np.asarray(p_lo, dtype=float)
        self.p_hi = None if p_hi is None else np.asarray(p_hi, dtype=float)
        self.stat = stat if stat else None
        self.aggregated = False if stat else True # Aggregation is deduced from stat declaration
   
    def _select_base(self, mode: Literal["values", "err", "p_lo", "p_hi"]) -> np.ndarray:
        if mode == "values":
            return self.values
        if mode == "err":
            if self.err is None:
                raise ValueError("No data available for mode='err'.")
            return self.err
        if mode == "p_lo":
            if self.p_lo is None:
                raise ValueError("No data available for mode='p_lo'.")
            return self.p_lo
        if mode == "p_hi":
            if self.p_hi is None:
                raise ValueError("No data available for mode='p_hi'.")
            return self.p_hi
        raise ValueError(f"Unsupported mode='{mode}'.")

    def _get_plot_data_and_edges(
        self, *, mode: Literal["values", "err", "p_lo", "p_hi"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (data_T, dose_edges_plot, let_edges_plot) without mutating self.
        Adds visual 0-padding for cumulative maps by duplicating the first row/col.
        """
        dose_edges = self.dose_edges.copy()
        let_edges = self.let_edges.copy()
        values = self._select_base(mode).copy()

        if self.cumulative:
            if dose_edges.size > 0 and dose_edges[0] > 0:
                dose_edges = np.insert(dose_edges, 0, 0.0)
                values = np.insert(values, 0, values[0, :], axis=0)
            if let_edges.size > 0 and let_edges[0] > 0:
                let_edges = np.insert(let_edges, 0, 0.0)
                values = np.insert(values, 0, values[:, 0], axis=1)

        return values.T, dose_edges, let_edges

    def plot(self, *, ax: Optional[plt.Axes] = None,
             cmap: str = None, colorbar: bool = True,
             mode: Literal["values", "err", "p_lo", "p_hi"] = "values",
             isovolumes: Optional[List[float]] = None,
             isovolumes_colors: Optional[Union[str, List[str]]] = None,
             interactive: bool = False, title: bool = False,
             auc_map: bool = False, **kwargs):

        data, dose_edges_plot, let_edges_plot = self._get_plot_data_and_edges(mode=mode)

        # Define custom colormap if not provided
        if not cmap:
            import seaborn as sns
            sns_cmap = sns.color_palette("Spectral", as_cmap=True)
            colors = [sns_cmap(i) for i in np.linspace(0.5, 1, 20)]
            cmap = ListedColormap(colors)

        # Setup figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=False)
        else:
            fig = ax.figure
        fig.subplots_adjust(bottom=0.6 if interactive else 0.12)

        mesh = ax.pcolormesh(dose_edges_plot, let_edges_plot, data, cmap=cmap, **kwargs)
        ax.set_xlabel(self.dose_label)
        ax.set_ylabel(self.let_label)
        if title: ax.set_title("Cumulative Dose–LET Volume Histogram (DLVH)")

        if self.cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if colorbar:
            cbar_label = "Volume [%]" if self.normalize else "Volume [cm³]"
            if auc_map: cbar_label = "AUC"
            cbar = plt.colorbar(mesh, ax=ax,
                         label=cbar_label,
                         aspect=10)
            cbar.ax.tick_params(size=0)

        # Helper: estimate total volume (needed if normalize=False) 
        def _total_volume(arr: np.ndarray) -> float:
            if self.cumulative:
                return float(arr[0, 0]) if arr.size > 0 else 0.0
            return float(np.nansum(arr))

        total_abs = _total_volume(self.values.T if mode == "values" else data)

        #  Draw static isovolumes
        if isovolumes:
            if self.normalize:
                levels_abs = list(isovolumes)  # already in %
            else:
                levels_abs = [p / 100.0 * total_abs for p in isovolumes]
            CS = ax.contour(
                dose_edges_plot[:-1],
                let_edges_plot[:-1],
                data,
                levels=levels_abs,
                colors=isovolumes_colors if isovolumes_colors else "black",
                linewidths=1,
        )
            fmt = (lambda v: f"{v:g}%") if self.normalize else (lambda v: f"{v:.2f}cm³")
            ax.clabel(CS, inline=True, fontsize=10, fmt=fmt)

        # Interactive slider for isovolume
        if interactive:
            # keep references on self to avoid GC
            self._fig2d = fig
            self._ax2d = ax
            self._data2d = data
            self._total_abs = total_abs
            self._interactive_contour = None
            self._interactive_labels = []
            self._dose_edges_plot = dose_edges_plot
            self._let_edges_plot = let_edges_plot

            ax_slider = plt.axes([0.15, 0.00001, 0.7, 0.04])
            self._slider = Slider(ax_slider,
                                  "Isovol [%]",
                                  valmin=0,
                                  valmax=100,
                                  valinit=0,
                                  valstep=1)

            def _update(_):
                # clear previous contour
                if self._interactive_contour is not None:
                    self._interactive_contour.remove()
                    self._interactive_contour = None
                for lbl in self._interactive_labels:
                    try:
                        lbl.remove()
                    except Exception:
                        pass
                self._interactive_labels = []

                level_pct = float(self._slider.val)
                if level_pct <= 0 or level_pct >= 100:
                    self._fig2d.canvas.draw_idle()
                    return

                if self.normalize:
                    level_abs = level_pct
                else:
                    level_abs = level_pct / 100.0 * self._total_abs

                self._interactive_contour = self._ax2d.contour(
                    self._dose_edges_plot[:-1],
                    self._let_edges_plot[:-1],
                    self._data2d,
                    levels=[level_abs],
                    colors="darkred",
                    linewidths=1.5
                )
                self._interactive_labels = self._ax2d.clabel(
                    self._interactive_contour,
                    inline=True,
                    fontsize=10,
                    fmt=lambda _: f"{level_pct:.0f}%"
                )
                self._fig2d.canvas.draw_idle()

            self._slider.on_changed(_update)

        if not ax: plt.show()
        return ax

    def get_marginals(self, *, quantity: Literal["dose", "let"] = "dose") -> Tuple[np.ndarray, np.ndarray]:
        """Return the marginal histogram as (edges, values). Only for cumulative 2D."""
        if not self.cumulative:
            raise NotImplementedError("Marginal extraction is only implemented for cumulative 2D histograms.")

        if quantity == "dose":
            edges = self.dose_edges.copy()
            values = self.values[:, 0].copy()
        elif quantity == "let":
            edges = self.let_edges.copy()
            values = self.values[0, :].copy()
        else:
            raise ValueError("Argument 'quantity' must be either 'dose' or 'let'.")

        # Step plotting convention
        if values.size == edges.size - 1:
            values = np.append(values, 0.0)
        elif values.size != edges.size:
            values = np.append(values, 0.0)

        return edges, values

    def plot_marginals(self, *, ax: Optional[plt.Axes] = None,
                       quantity: Literal["dose", "let"] = "dose", **kwargs):
        """Plot DVH or LVH derived from the cumulative 2D histogram."""
        edges, values = self.get_marginals(quantity=quantity)

        if ax is None:
            _, ax = plt.subplots()

        if quantity == "dose":
            ax.set_xlabel(self.dose_label)
            ax.set_title("DVH from 2D cumulative")
        else:
            ax.set_xlabel(self.let_label)
            ax.set_title("LVH from 2D cumulative")
        ax.step(edges, values, where="post", **kwargs)

        ax.set_ylabel("Volume [%]" if self.normalize else "Volume [cm³]")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        return ax


class DLVH:
    """
    Dose–LET Volume Histogram (DLVH).

    You may provide only `dose` (for DVH) or only `let` (for LVH),
    or both (to compute DVH, LVH and 2D DLVH).
    """

    def __init__(self, *,
                 dose: Optional[np.ndarray] = None,
                 let: Optional[np.ndarray] = None,
                 volume_cc: float,
                 volume_fraction: float = 100.,
                 relative_volumes: Optional[np.ndarray] = None,
                 dose_units: str = "Gy",
                 let_units: str = "keV/µm"):
        if dose is None and let is None:
            raise ValueError("At least one between dose or let must be provided.")

        self.dose = self._validate_array(dose, "dose") if dose is not None else None
        self.let = self._validate_array(let, "let") if let is not None else None

        if self.dose is not None and self.let is not None:
            if self.dose.shape != self.let.shape:
                raise ValueError("Dose and LET arrays must have the same shape.")

        if volume_cc <= 0 or not np.isfinite(volume_cc):
            raise ValueError("Volume must be a positive finite value (cm³).")
        self.volume_cc = float(volume_cc)

        size = self.dose.size if self.dose is not None else self.let.size
        if relative_volumes is None:
            relw = np.full(size, 1.0 / size, dtype=float)
        else:
            relw = np.asarray(relative_volumes, dtype=float).ravel()
            if relw.shape[0] != size:
                raise ValueError("relative_volumes must match dose/let length.")
            if np.any(~np.isfinite(relw)) or np.any(relw < 0):
                raise ValueError("relative_volumes must be finite and non-negative.")
            sumw = float(relw.sum())
            if sumw <= 0:
                raise ValueError("Sum of relative_volumes must be > 0.")
            relw = relw / sumw
        self.relw = relw

        self.n_voxels = size
        self.dose_units = dose_units
        self.let_units = let_units
        self.volume_fraction = volume_fraction

    @staticmethod
    def _validate_array(arr: Optional[np.ndarray], label: str) -> np.ndarray:
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            raise ValueError(f"{label} array cannot be empty.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{label} array must contain non-negative finite values.")
        return arr

    def _dose_at_volume(self, *, data: np.ndarray,
                        weights: np.ndarray,
                        volume_cc: float,
                        volume_grid: np.ndarray,
                        normalize: Optional[bool])->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Sort data and weights according to data order
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Compute cumulative volume
        cumulative_volume = np.cumsum(sorted_weights[::-1])[::-1]
        # Normalization
        if normalize:
            cumulative_volume = (cumulative_volume / volume_cc) * 100.0

        # Invert DVH: compute D = f(V)
        dose_grid = np.interp(volume_grid, cumulative_volume[::-1], sorted_data[::-1])

        edges = dose_grid[::-1]
        values = np.max(volume_grid) - volume_grid
        centers = _get_bin_centers(edges=edges)
        
        return centers, values, edges
    
    def _volume_histogram(self, *, data: np.ndarray, weights: np.ndarray,
                          quantity: str,
                          bin_centers: Optional[np.ndarray] = None,
                          bin_edges: Optional[np.ndarray] = None,
                          bin_width: Optional[float] = None,
                          normalize: bool = True,
                          cumulative: bool = True,
                          aggregatedby: str = None) -> Histogram1D:
        
        if aggregatedby not in [None, "dose", "volume", "let"]:
            raise ValueError("Unsupported aggregateby. Choose 'dose', 'let' or 'volume'.")
        if aggregatedby == "volume" and not cumulative:
            raise ValueError(f"Unsupported sampling ({aggregatedby}) for differential DVH.")
        volume_binning = aggregatedby == "volume" or (aggregatedby == None and cumulative)
        data_binning = aggregatedby == "dose" or aggregatedby == "let" or (aggregatedby == None and not cumulative)
        if bin_centers is not None and len(bin_centers) < 2:
                raise ValueError("At least two data_bin_centers elements are required to compute a volume histogram.")
        
        # Dose/let binning
        if data_binning:
            # Bin centers provided for dose/let
            if bin_centers is not None:
                # Get dose/let edges
                centers = bin_centers
                edges = _get_bin_edges(centers=bin_centers)

            # Bin width provided for dose/let  
            elif bin_width is not None:
                xmax = float(np.max(data))
                n_bins = int(np.ceil(xmax / bin_width)) if bin_width > 0 else 1
                edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
                centers = _get_bin_centers(edges=edges)

            # Bin edges provided for dose/let  
            elif bin_edges is not None:
                edges = bin_edges
                centers = _get_bin_centers(edges=bin_edges)

            else: # default binning: dose/let binning
                edges = _auto_bins(array=data)
                centers = _get_bin_centers(edges=edges)
            # Compute corresponding volumes
            volumes, _ = np.histogram(data, bins=edges, weights=weights)

            # Cumulative distribution
            if cumulative:
                volumes = np.cumsum(volumes[::-1])[::-1]
                volumes = np.append(volumes, 0.0) # Append last zero bin
            # Normalization
            if normalize:
                volumes = (volumes / self.volume_cc) * 100.0
            values = volumes.astype(float)

        # Volume binning
        elif volume_binning:

            if not cumulative:
                raise ValueError("Volume binning is only supported for cumulative histograms.")

            # Bin centers provided for volume
            if bin_centers is not None:
                centers, values, edges = self._dose_at_volume(data=data,
                                                              weights=weights,
                                                              volume_cc=self.volume_cc,
                                                              volume_grid=bin_centers,
                                                              normalize=normalize)

            # Bin width provided for volume
            elif bin_width is not None:
                vol_max = 100.0 if normalize else self.volume_cc
                n_bins = int(np.ceil(vol_max / bin_width)) if bin_width > 0 else 1
                volume_edges = np.linspace(0.0, (1+n_bins) * bin_width, n_bins + 1)
                volume_centers = (volume_edges[:-1] + volume_edges[1:]) / 2.0
                centers, values, edges = self._dose_at_volume(data=data,
                                                              weights=weights,
                                                              volume_cc=self.volume_cc,
                                                              volume_grid=volume_centers,
                                                              normalize=normalize)

            # Bin edges provided for volume
            elif bin_edges is not None:
                raise ValueError("Volume binning providing bin edges is not available.")

            # default binning + cumulative: volume binning
            else:
                center_step = 0.01 if normalize else 0.01 # 0.01% or 0.01 cc step
                max_center = 100 * self.volume_fraction if normalize else self.volume_cc * self.volume_fraction
                max_volume = max_center + center_step/2
                volume_centers = np.arange(0, max_volume, center_step) if normalize else np.arange(0, max_volume, center_step) 
                centers, values, edges = self._dose_at_volume(data=data,
                                                              weights=weights,
                                                              volume_cc=self.volume_cc,
                                                              volume_grid=volume_centers,
                                                              normalize=normalize)

        x_label = f"Dose [{self.dose_units}]" if quantity == "dose" else f"LET [{self.let_units}]"
        return Histogram1D(values=values, edges=edges, centers=centers,
                           quantity=quantity, normalize=normalize,
                           cumulative=cumulative, x_label=x_label, aggregatedby=aggregatedby)

    def dose_volume_histogram(self, *, bin_width: Optional[float] = None,
                              bin_centers: Optional[np.ndarray] = None,
                              bin_edges: Optional[np.ndarray] = None,
                              normalize: bool = True,
                              cumulative: bool = True,
                              let_threshold: float = 0.0,
                              aggregatedby: str = None) -> Histogram1D:
        if self.dose is None:
            raise ValueError("Dose array not available for DVH.")
        if let_threshold > 0 and self.let is None:
            raise ValueError("LET array required to apply let_threshold.")
        if aggregatedby not in [None, "dose", "volume"]:
            raise ValueError("Unsupported aggregateby. Choose 'dose' or 'volume'.")

        mask = np.ones_like(self.dose, dtype=bool) if self.let is None else self.let >= let_threshold
        data = self.dose[mask]
        weights = (self.relw * self.volume_cc)[mask]
        self.volume_fraction = np.sum(self.relw[mask])

        return self._volume_histogram(data=data, weights=weights, quantity="dose",
                                      bin_width=bin_width, bin_centers=bin_centers, 
                                      bin_edges=bin_edges, normalize=normalize, 
                                      cumulative=cumulative, aggregatedby=aggregatedby)

    def let_volume_histogram(self, *, bin_width: Optional[float] = None,
                             bin_centers: Optional[np.ndarray] = None,
                             bin_edges: Optional[np.ndarray] = None,
                             normalize: bool = True,
                             cumulative: bool = True,
                             dose_threshold: float = 0.0,
                             aggregatedby: str = None) -> Histogram1D:
        if self.let is None:
            raise RuntimeError("LET array not available for LVH.")
        if dose_threshold > 0 and self.dose is None:
            raise RuntimeError("Dose array required to apply dose_threshold.")
        if aggregatedby not in [None, "let", "volume"]:
            raise ValueError("Unsupported aggregateby. Choose 'let' or 'volume'.")

        mask = np.ones_like(self.let, dtype=bool) if self.dose is None else self.dose >= dose_threshold
        data = self.let[mask]
        weights = (self.relw * self.volume_cc)[mask]
        self.volume_fraction = np.sum(self.relw[mask])

        return self._volume_histogram(data=data, weights=weights, quantity="let",
                                      bin_width=bin_width, bin_centers=bin_centers, 
                                      bin_edges=bin_edges, normalize=normalize, 
                                      cumulative=cumulative, aggregatedby=aggregatedby)

    def dose_let_volume_histogram(self, *, bin_width_dose: Optional[float] = None,
                                  bin_width_let: Optional[float] = None,
                                  dose_edges: Optional[np.ndarray] = None,
                                  let_edges: Optional[np.ndarray] = None,
                                  normalize: bool = True,
                                  cumulative: bool = True) -> Histogram2D:
        if self.dose is None or self.let is None:
            raise RuntimeError("Both dose and LET arrays are required for 2D DLVH.")

        # dose edges
        if dose_edges is not None:
            d_edges = np.asarray(dose_edges, dtype=float)
            if np.max(d_edges) < np.max(self.dose):
                raise ValueError("Provided dose_edges do not cover the maximum dose value.")
        elif bin_width_dose is None:
            d_edges = _auto_bins(array=self.dose)
        else:
            dmax = float(np.max(self.dose))
            nd = int(np.ceil(dmax / bin_width_dose))
            d_edges = np.linspace(0.0, nd * bin_width_dose, nd + 1)

        # let edges
        if let_edges is not None:
            l_edges = np.asarray(let_edges, dtype=float)
            if np.max(l_edges) < np.max(self.let):
                raise ValueError("Provided l_edges do not cover the maximum dose value.")
        elif bin_width_let is None:
            l_edges = _auto_bins(array=self.let)
        else:
            lmax = float(np.max(self.let))
            nl = int(np.ceil(lmax / bin_width_let))
            l_edges = np.linspace(0.0, nl * bin_width_let, nl + 1)

        weights = self.relw * self.volume_cc

        vols, d_edges, l_edges = np.histogram2d(self.dose, self.let,
                                                bins=(d_edges, l_edges),
                                                weights=weights)

        values = _suffix_cumsum2d(vols) if cumulative else vols.astype(float)

        if normalize:
            values = (values / self.volume_cc) * 100.0
            
        return Histogram2D(values=values,
                           dose_edges=d_edges,
                           let_edges=l_edges,
                           normalize=normalize,
                           cumulative=cumulative,
                           dose_label=f"Dose [{self.dose_units}]",
                           let_label=f"LET [{self.let_units}]")
