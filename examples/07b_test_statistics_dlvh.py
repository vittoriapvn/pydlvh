"""
07b_test_statistics_dlvh.py
============================
Compare two DLVH cohorts applying different statistical
tests.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pydlvh.core import DLVH
from pydlvh.utils import _get_bin_centers
import pydlvh.analyzer as analyzer

def create_synthetic_patient(n_voxels=4000,
                             mu_dose=30.0, sigma_dose=7.0,
                             mu_let=30.0, sigma_let=1.0,
                             volume_rng=(80.0, 120.0)):
    """
        Create synthetic dose, let and relative volumes distributions.
        Dose is uniform in [0, dose_max], LET is Gaussian troncated for 
        values >=0, the relative weights are Gaussian.
    """
    dose = np.random.normal(loc=mu_dose, scale=sigma_dose, size=n_voxels)
    dose = np.clip(dose, 0.0, None)
    let = np.random.normal(loc=mu_let, scale=sigma_let, size=n_voxels)
    let = np.clip(let, 0.0, None)

    relative_volumes = np.exp(-0.5 * ((dose - mu_dose) / max(sigma_dose, 1e-6))**2)
    if not np.any(relative_volumes > 0):
        relative_volumes[:] = 1.0
    relative_volumes = relative_volumes / relative_volumes.sum()

    volume_cc = np.random.uniform(*volume_rng)
    
    return DLVH(dose=dose, let=let, volume_cc=volume_cc, relative_volumes=relative_volumes)

def main():
    np.random.seed(7)

    # 1) Create synthetic control and ae cohorts
    mu_dose_control, sigma_dose_control = 60.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_control, scale=3.0, size=10), np.random.normal(loc=sigma_dose_control, scale=1.0, size=100))]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    mu_dose_ae, sigma_dose_ae = 50.0, 5.0 
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_ae, scale=3.0, size=10), np.random.normal(loc=sigma_dose_ae, scale=1.0, size=100))]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group

    # 2) Median DLVHs
    # Set manual uniform dose+let binning for aggregation
    dose_edges = np.arange(0., 96, 1)  # D in [0, 95] with step 1 Gy
    let_edges = np.arange(0., 51, 1)  # LET in [0, 50] with step 1 keV/um
    print("\nAggregating control DLVHs...")
    all_control_dlvhs, median_control_dlvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                                stat="median",
                                                                quantity="dlvh",
                                                                dose_edges=dose_edges,
                                                                let_edges=let_edges)
    print("\nAggregating AE DLVHs...")
    all_ae_dlvhs, median_ae_dlvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                      stat="median",
                                                      quantity="dlvh",
                                                      dose_edges=dose_edges,
                                                      let_edges=let_edges)

    # 3) Compute statistical significance between control and AE DLVHs (Mann-Whitney u-test)
    alpha = 0.05
    print("\nComputing voxel-based Mann-Whitney test...")
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")

    # Print (top 5 most) significant Dx% according to Mann-Whitney U-test
    if np.any(significance):
        rows = []
        significant_indices = np.argwhere(significance)
        for i, j in significant_indices:
            rows.append({
                "Dose (Gy)": median_control_dlvh.dose_edges[i],
                "LET (keV/µm)": median_control_dlvh.let_edges[j],
                "Volume control": median_control_dlvh.values[i, j],
                "Volume AE": median_ae_dlvh.values[i, j],
                "p-value": pvalues[i, j],
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("p-value").head(5)
        print(f"\nTop 5 Mann–Whitney U-test most significant Dx% (α={alpha}, BH corrected):\n")
        print(df.to_markdown(index=False, floatfmt=".4g"))

    # 4) Plot median DLVHs
    _, ax = plt.subplots(1, 2, figsize=(9, 4))
    isovolumes_colors = [ "#21918c", "#DF0E6F", "#fde725"]
    median_control_dlvh.plot(ax=ax[0], isovolumes=[20, 50, 80], isovolumes_colors=isovolumes_colors)
    ax[0].set_title("Median Control DLVH")
    median_ae_dlvh.plot(ax=ax[1], isovolumes=[20, 50, 80], isovolumes_colors=isovolumes_colors)
    ax[1].set_title("Median AE DLVH")
    plt.tight_layout()
    plt.show()

    # 5) Compute AUC map between control and AE DLVHs    
    print("\nComputing ROC-AUC score...")
    auc_map = analyzer.get_auc_score(control_histograms=all_control_dlvhs,
                                     ae_histograms=all_ae_dlvhs)
    
    # Print (top 5 most) significant Dx% according to AUC score
    auc_threshold = 0.8
    significance = auc_map.values > auc_threshold
    if np.any(significance):
        rows = []
        significant_indices = np.argwhere(significance)
        for i, j in significant_indices:
            rows.append({
                "Dose (Gy)": median_control_dlvh.dose_edges[i],
                "LET (keV/µm)": median_control_dlvh.let_edges[j],
                "Volume control": median_control_dlvh.values[i, j],
                "Volume AE": median_ae_dlvh.values[i, j],
                "AUC score": auc_map.values[i, j],
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("AUC score", ascending=False).head(5)
        print(f"\nTop 5 most significant AUC scores:\n")
        print(df.to_markdown(index=False, floatfmt=".4g"))
    
    # 6) Plot AUC map and visualize signficant voxels
    _, ax = plt.subplots(figsize=(5, 4))
    auc_map.plot(ax=ax, auc_map=True)
    dose_centers = _get_bin_centers(edges=auc_map.dose_edges)
    let_centers = _get_bin_centers(edges=auc_map.let_edges)
    ax.contour(
        dose_centers, let_centers, significance.T,
        levels=[0.5], colors="darkred", linewidths=2
    )
    ax.set_title("Voxel-wise AUC (Control vs AE)")
    legend_elements = [Line2D([0], [0], marker='', linestyle='-', 
                              color="darkred", linewidth=2,
                              label=" AUC > 0.8")]
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, handletextpad=0.14)
    plt.show()

if __name__ == "__main__":
    main()
