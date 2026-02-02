from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Literal, Optional, Union
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

from pydlvh.core import Histogram1D, Histogram2D, DLVH
from .utils import suggest_common_edges, suggest_common_edges_2d, _get_bin_edges

""" 
    Methods to analyze DVH/LVH/DLVH cohorts. The methods either extract the statistics
    of histogram cohorts or compare control and adverse event (ae) groups to investigate 
    possible statistical difference.
"""
        
def validate(histograms: Union[Histogram1D, Histogram2D, List[Union[Histogram1D, Histogram2D]]],
             validate_edges: bool = False):

    """ Check that all the histograms belong to the same cohort. """

    if isinstance(histograms, list):
    
        reference_histogram = histograms[0]
        reference_type = type(reference_histogram)

        for h in histograms[1:]:
            if type(h) != reference_type:
                raise ValueError("All histograms must be of the same type (1D or 2D).")
            
            if isinstance(reference_histogram, Histogram2D) and validate_edges:
                if not (np.array_equal(reference_histogram.dose_edges, h.dose_edges) and
                        np.array_equal(reference_histogram.let_edges, h.let_edges)):
                    raise ValueError("All 2D histograms must have matching dose and LET edges.")
            
            elif isinstance(reference_histogram, Histogram1D):
                # Validate either dose/let edges (x sampling) or volume edges (y sampling)
                if validate_edges:
                    if not np.array_equal(reference_histogram.edges, h.edges) or not np.array_equal(reference_histogram.values, h.values) or not np.array_equal(reference_histogram.centers, h.centers):
                        raise ValueError("All 1D histograms must have matching edges either on the dose/let or the volumes axis.")
                    
                # Validate quantity type
                if reference_histogram.quantity != h.quantity:
                    raise ValueError("All 1D histograms must contain the same quantity type ('dvh' or 'lvh').")
                
                # If aggregated, validate aggregation modality
                if hasattr(reference_histogram, 'aggregatedby') and hasattr(h, 'aggregatedby'):
                    if reference_histogram.aggregatedby != h.aggregatedby:
                        raise ValueError("If aggregated, all 1D histograms must be aggregated by the same modality ('dose', 'let' or 'volume').")

def get_quantity_at_volume(histo: Histogram1D,
                           volumes: np.ndarray) -> np.ndarray:
    
    """ Interpolate histogram at specified volumes to get dose/let at given volumes. """

    quantities = np.interp(
        volumes,
        histo.values[::-1],
        histo.edges[::-1]
    )

    return quantities

def build_statistics_matrix(control_histograms, ae_histograms,
                            dose_at_volumes: Optional[np.ndarray] = None,
                            fill_value: float = 1.0,
                            test: str = ["Mann-Whitney", "Wilcoxon"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    is_aggregated = hasattr(reference_histogram, 'aggregatedby') and reference_histogram.aggregatedby is not None

    if histo_type == Histogram1D: # Histogram1D
        if is_aggregated:
            if dose_at_volumes is not None:
                # Interpolate histograms at specified dose_at_volumes
                control_group = np.stack([get_quantity_at_volume(histo=histo, volumes=dose_at_volumes) for histo in control_histograms])
                ae_group = np.stack([get_quantity_at_volume(histo=histo, volumes=dose_at_volumes) for histo in ae_histograms])
            elif reference_histogram.aggregatedby == "volume":
                # Invert histograms and stack
                control_group = np.stack([histo.edges for histo in control_histograms])
                ae_group = np.stack([histo.edges for histo in ae_histograms])
            elif reference_histogram.aggregatedby in ["dose", "let"]:
                # Just stack histograms
                control_group = np.stack([histo.values for histo in control_histograms])
                ae_group = np.stack([histo.values for histo in ae_histograms])
            else:
                raise ValueError(f"For Histogram1D {test} test, histograms must be aggregated by 'volume', 'dose' or 'let'.")
        
        else:
            # Default option: assuming aggregation by volume
            # Invert histograms
            control_group = np.stack([histo.edges for histo in control_histograms])
            ae_group = np.stack([histo.edges for histo in ae_histograms])

        shape = (control_group.shape[1])
        original_stats_matrix = np.full(shape, fill_value, dtype=float)
    
    else: # Histogram2D
        # Stack histogram values 
        control_group = np.stack([histo.values for histo in control_histograms])
        ae_group = np.stack([histo.values for histo in ae_histograms])
        shape = (control_group.shape[1], control_group.shape[2])
        original_stats_matrix = np.full(shape, fill_value, dtype=float)

    return control_group, ae_group, original_stats_matrix, shape

def aggregate(dlvhs: Union[DLVH, List[DLVH]],
              quantity: Optional[Literal["dvh", "lvh", "dlvh"]] = None,
              aggregateby: Optional[Literal["volume", "dose", "let"]] = None,
              stat: Literal["mean", "median"] = "median",
              dose_edges: Optional[np.ndarray] = None,
              let_edges: Optional[np.ndarray] = None,
              centers: Optional[np.ndarray] = None,
              normalize: bool = True,
              cumulative: bool = True,
              dose_units: str = "Gy(RBE)",
              let_units: str = "keV/µm") -> Tuple[Union[Histogram1D, Histogram2D], Union[Histogram1D, Histogram2D]]:

    """Aggregate DVH/LVH/DLVH across a cohort."""

    quantity = quantity or "dlvh"
    if quantity not in ["dvh", "lvh", "dlvh"]:
        raise ValueError(f"Unrecognized quantity '{quantity}'. Please select 'dvh', 'lvh', or 'dlvh'.")
    is_1D = quantity in ("dvh", "lvh")
    is_2D = quantity == "dlvh"

    # Validate aggregateby
    valid_aggregateby = ["dose", "let", "volume", None]
    if aggregateby not in valid_aggregateby:
        raise ValueError(f"Unsupported aggregateby '{aggregateby}'. Choose 'dose', 'let', or 'volume'.")
    if aggregateby == "volume" and cumulative == False:
        raise ValueError(f"Unsupported aggregate sampling ({aggregateby}) for differential DVHs.")
    if aggregateby == "dose" and quantity == "lvh":
        raise ValueError(f"Cannot aggregate LVHs by dose.")
    if aggregateby == "let" and quantity == "dvh":
        raise ValueError(f"Cannot aggregate DVHs by let.")

    if is_1D:
        # Set default aggregateby
        if cumulative: aggregateby = aggregateby or ("volume")
        else: 
            if aggregateby is None:
                aggregateby = "dose" if quantity == "dvh" else "let"

        # Select edges for binning
        if aggregateby == "dose":
            x_edges = dose_edges
        elif aggregateby == "let":
            x_edges = let_edges
        else:
            x_edges = None
        y_edges = None
    elif is_2D:
        x_edges = dose_edges
        y_edges = let_edges

    # Normalize input to list of DLVHs (also for single DLVH provided)
    dlvhs = normalize_to_list(dlvhs)

    # Get rebinned cohort histograms
    rebinned_histos, bin_edges = get_all_cohort_histograms(
        dlvhs=dlvhs,
        x_edges=x_edges,
        y_edges=y_edges,
        bin_centers=centers,
        quantity=quantity,
        aggregateby=aggregateby,
        cumulative=cumulative,
        normalize=normalize
    )

    # Compute statistics
    if is_1D and aggregateby == "volume":
        stack = np.stack([h.edges for h in rebinned_histos], axis=0)
    else:
        stack = np.stack([h.values for h in rebinned_histos], axis=0)

    if stat == "mean":
        aggregate = np.mean(stack, axis=0)
        error = np.std(stack, axis=0, ddof=1)
        lower_percentile = higher_percentile = None
    elif stat == "median":
        aggregate = np.median(stack, axis=0)
        lower_percentile = np.percentile(stack, q=25, axis=0)
        higher_percentile = np.percentile(stack, q=75, axis=0)
        error = None
    else:
        raise ValueError("Unsupported stat. Choose 'mean' or 'median'.")

    if aggregateby == "volume":
        values = rebinned_histos[0].values
        edges = aggregate
    else:
        values = aggregate
        edges = bin_edges

    # Assign correct label if needed
    dose_label = f"Dose [{dose_units}]"
    let_label = rf"LET$_d$ [{let_units}]"

    if is_1D:
        x_label = dose_label if quantity == "dvh" else let_label
        return rebinned_histos, Histogram1D(values=values, edges=edges,
                                            quantity=quantity, normalize=normalize,
                                            cumulative=cumulative, x_label=x_label,
                                            err=error, p_lo=lower_percentile, p_hi=higher_percentile,
                                            stat=stat, aggregatedby=aggregateby)
    elif is_2D:
        return rebinned_histos, Histogram2D(values=values,
                                            dose_edges=edges[0], let_edges=edges[1],
                                            normalize=normalize, cumulative=cumulative,
                                            dose_label=dose_label, let_label=let_label,
                                            err=error, p_lo=lower_percentile, p_hi=higher_percentile,
                                            stat=stat)
    
    return None

def aggregate_marginals(
        dlvhs: Union[DLVH, List[DLVH]],
        quantity: Literal["dose", "let"] = "median",
        stat: Literal["mean", "median"] = "median",
        normalize: bool = True,
        cumulative: bool = True,
        units: str = None
    ) -> Histogram1D:

    """
        Aggregate DVH or LVH derived from the 2D DLVH aggregate.
    """

    # If not already, aggregate input
    if not isinstance(dlvhs, list) and dlvhs.aggregated:
        aggregated_histo = dlvhs
    else: 
        aggregated_histo = aggregate(dlvhs=dlvhs, stat=stat, normalize=normalize, cumulative=cumulative)

    edges, values = aggregated_histo.get_marginals(quantity=quantity)
    if not units: units = "Gy[RBE]" if quantity == "dose" else f"keV/$\mu$m"
    label = f"{quantity} [{units}]"
    return Histogram1D(values=values, edges=edges,
                       quantity=quantity, normalize=normalize,
                       cumulative=cumulative, x_label=label,
                       err=aggregated_histo.err[:, 0] if stat == "mean" and aggregated_histo.err is not None else None,
                       p_lo=aggregated_histo.p_lo[:, 0] if stat == "median" and aggregated_histo.p_lo is not None else None,
                       p_hi=aggregated_histo.p_hi[:, 0] if stat == "median" and aggregated_histo.p_hi is not None else None)

def normalize_to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def compute_2d_edges(dlvhs: List[DLVH],
                     dose_edges: Optional[np.ndarray] = None,
                     let_edges: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    if dose_edges is not None and let_edges is not None:
        raise ValueError("compute_2d_edges is intended for missing edeges only.")
    
    if dose_edges is not None: # Only dose provided
        print("Only dose binning has been specified for aggregated dlvh. Let binning will be atuomatically set.")
        lets = [dlvh.let for dlvh in dlvhs] 
        let_edges = suggest_common_edges(arrays=lets)

    elif let_edges is not None: # Only let provided
        print("Only let binning has been specified for aggregated dlvh. Dose binning will be atuomatically set.")
        doses = [dlvh.dose for dlvh in dlvhs] 
        dose_edges = suggest_common_edges(arrays=doses)

    else: # None specified
        doses = [dlvh.dose for dlvh in dlvhs]
        lets = [dlvh.let for dlvh in dlvhs] 
        dose_edges, let_edges = suggest_common_edges_2d(dose_arrays=doses, let_arrays=lets)

    return dose_edges, let_edges

def build_histogram(dlvh,
                    quantity: Literal["dvh", "lvh", "dlvh"],
                    edges: np.ndarray,
                    centers: np.ndarray,
                    cumulative: bool,
                    normalize: bool,
                    aggregateby: Optional[Literal["volume", "dose", "let"]] = None):

    if quantity == "dvh":
        return dlvh.dose_volume_histogram(
            bin_edges=edges,
            bin_centers=centers,
            cumulative=cumulative,
            normalize=normalize,
            aggregatedby=aggregateby,
        )

    if quantity == "lvh":
        return dlvh.let_volume_histogram(
            bin_edges=edges,
            bin_centers=centers,
            cumulative=cumulative,
            normalize=normalize,
            aggregatedby=aggregateby,
        )

    if quantity == "dlvh":
        dose_edges, let_edges = edges
        return dlvh.dose_let_volume_histogram(
            dose_edges=dose_edges,
            let_edges=let_edges,
            cumulative=cumulative,
            normalize=normalize,
        )

    raise ValueError(f"Unsupported histogram quantity: {quantity}")

def get_all_cohort_histograms(
        dlvhs: Union[DLVH, List[DLVH]],
        bin_centers: Optional[np.ndarray] = None, # binning
        x_edges: Optional[np.ndarray] = None, # binning along the first x-axis (D)
        y_edges: Optional[np.ndarray] = None, # binning along the first y-axis (L)
        quantity: Literal["dvh", "lvh", "dlvh"] = None,
        aggregateby: Optional[Literal["volume", "dose", "let"]] = None,
        cumulative: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:

    """ Return an array containing all the DVH/LVH/DLVHs from cohort after standardizing binning. """

    dlvhs = normalize_to_list(dlvhs)

    quantity = quantity or "dlvh" # default quantity: dlvh
    if quantity not in ["dvh", "lvh", "dlvh"]:
        raise ValueError(f"Unrecognized {quantity}. Please select 'dvh', 'lvh', or 'dlvh'.")
    edges = None
    centers = None

    # 1D case
    if quantity in ("dvh", "lvh"):
        if bin_centers is not None:
            centers = bin_centers
        elif x_edges is not None:
            edges = x_edges
        else:
            if aggregateby in ["dose", "let"]:
                arrays = [dlvh.dose if quantity == "dvh" else dlvh.let for dlvh in dlvhs]
                edges = suggest_common_edges(arrays=arrays)
            else: # aggregatedby == "volume"
                if cumulative:
                    pass
                else: raise ValueError(f"Unsupported sampling ({aggregateby}) for differential DVHs.")
    
    # 2D case
    else:
        if centers is not None:
            raise ValueError("To create cohort DLVHs, centers do not need to be specified.")
        
        if x_edges is not None and y_edges is not None:
            dose_edges, let_edges = x_edges, y_edges
        else: 
            dose_edges, let_edges = compute_2d_edges(dlvhs, x_edges, y_edges)
        edges = (dose_edges, let_edges)

    rebinned_histos = [
        build_histogram(dlvh=dlvh, quantity=quantity, edges=edges, centers=centers, aggregateby=aggregateby, cumulative=cumulative, normalize=normalize)
        for dlvh in dlvhs
    ]

    return rebinned_histos, edges
    
def voxel_wise_Mann_Whitney_test(control_histograms: List[Union[Histogram1D, Histogram2D]],
                                 ae_histograms: List[Union[Histogram1D, Histogram2D]],
                                 alpha: float = 0.05,
                                 dose_at_volumes: Optional[np.ndarray] = None,
                                 alternative: Optional[Literal["two-sided", "greater", "less"]] = "two-sided",
                                 correction: Optional[Literal["holm", "fdr_bh"]] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """ Perform voxel-wise Mann-Whitney U test between control and ae groups. """

    validate([*control_histograms, *ae_histograms])

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    control_group, ae_group, original_p_values, shape = build_statistics_matrix(control_histograms,
                                                                                ae_histograms, 
                                                                                dose_at_volumes=dose_at_volumes,
                                                                                fill_value=1.0,
                                                                                test="Mann-Whitney")

    # Perform Mann-Whitney u test
    for idx in np.ndindex(shape):
        if histo_type == Histogram1D:
            control = control_group[:, idx[0]]
            ae = ae_group[:, idx[0]]
        else: # Histogram2D
            control = control_group[:, idx[0], idx[1]]
            ae = ae_group[:, idx[0], idx[1]]
        _, p = mannwhitneyu(control, ae, alternative=alternative)
        original_p_values[idx] = p

    original_p_values = original_p_values.flatten()

    # Apply test correction
    if correction:
        reject, p_values, _, _ = multipletests(original_p_values, alpha=alpha, method=correction)
    else:
        reject = original_p_values < alpha
        p_values = original_p_values

    p_values = p_values.reshape(shape)
    significance = reject.reshape(shape)

    return p_values, significance

def voxel_wise_Wilcoxon_test(control_histograms: List[Union[Histogram1D, Histogram2D]],
                             ae_histograms: List[Union[Histogram1D, Histogram2D]], 
                             dose_at_volumes: Optional[np.ndarray] = None,
                             alpha: float = 0.05,
                             alternative: Optional[Literal["two-sided", "greater", "less"]] = "two-sided",
                             correction: Optional[Literal["holm", "fdr_bh"]] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """ Perform voxel-wise Wilcoxon test between control and ae groups. """

    validate([*control_histograms, *ae_histograms])

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    control_group, ae_group, original_p_values, shape = build_statistics_matrix(control_histograms,
                                                                                ae_histograms, 
                                                                                dose_at_volumes=dose_at_volumes,
                                                                                fill_value=1.0,
                                                                                test="Wilcoxon")

    # Perform Mann-Whitney u test
    for idx in np.ndindex(shape):
        if histo_type == Histogram1D:
            control = control_group[:, idx[0]]
            ae = ae_group[:, idx[0]]
        else: # Histogram2D
            control = control_group[:, idx[0], idx[1]]
            ae = ae_group[:, idx[0], idx[1]]
            if np.all(control == ae):
                p = 1.0
            else:
                _, p = wilcoxon(control, ae, alternative=alternative)
        original_p_values[idx] = p

    original_p_values = original_p_values.flatten()

    # Apply test correction
    if correction:
        reject, p_values, _, _ = multipletests(original_p_values, alpha=alpha, method=correction)
    else:
        reject = original_p_values < alpha
        p_values = original_p_values

    p_values = p_values.reshape(shape)
    significance = reject.reshape(shape)

    return p_values, significance

def get_auc_score(control_histograms: Union[List[Histogram1D], List[Histogram2D]],
                  ae_histograms: Union[List[Histogram1D], List[Histogram2D]],
                  dose_at_volumes: Optional[np.ndarray] = None)  -> Histogram2D:
    
    validate([*control_histograms, *ae_histograms])

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    control_group, ae_group, auc_map, shape = build_statistics_matrix(control_histograms,
                                                                      ae_histograms, 
                                                                      fill_value=0.5,
                                                                      test="AUC score",
                                                                      dose_at_volumes=dose_at_volumes)

    for idx in np.ndindex(shape):

        control_value = control_group[:, idx[0], idx[1]]
        ae_value = ae_group[:, idx[0], idx[1]]

        y_true = np.array([0]*len(ae_value) + [1]*len(control_value))
        y_scores = np.concatenate([ae_value, control_value])

        # Only compute AUC if there is variation in data
        if np.unique(y_scores).size > 1:
            auc_map[idx] = roc_auc_score(y_true, y_scores)
        else:
            auc_map[idx] = 0.5  # no discrimination
    
    return Histogram2D(values=auc_map,
                       dose_edges=reference_histogram.dose_edges if histo_type == Histogram2D else None,
                       let_edges=reference_histogram.let_edges if histo_type == Histogram2D else None,
                       cumulative=reference_histogram.cumulative if histo_type == Histogram2D else None,
                       normalize=reference_histogram.normalize if histo_type == Histogram2D else None,
                       dose_label=reference_histogram.dose_label if histo_type == Histogram2D else None,
                       let_label=reference_histogram.let_label if histo_type == Histogram2D else None)