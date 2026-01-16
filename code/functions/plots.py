import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout

from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

CHANNEL_ORDER = ["C3", "F4", "F3", "Fp2", "Fp1", "Pz", "Cz", "Fz", "T6", "T5", "T4", "T3", "F8", "F7", "O2", "O1", "P4", "P3", "C4"]
AREA_COLORS = {
    "Fp": "#E41A1C",
    "F":  "#4DAF4A",
    "C":  "#377EB8",
    "P":  "#984EA3",
    "O":  "#FF7F00",
    "T":  "#A65628",
}

def channel_color(ch):
    if ch.startswith("Fp"):
        return AREA_COLORS["Fp"]
    return AREA_COLORS[ch[0]]


def plot_ecn(
    mean_pvals,
    binary_adj,
    channel_order,
    threshold=1.3,
    title="ECN",
    figsize=(8, 8)
):
    """
    Plot ECN using MNE connectivity circle.
    """
    conn = prepare_mne_connectivity(mean_pvals, binary_adj, channel_order)

    node_colors = [channel_color(ch) for ch in channel_order]

    plot_connectivity_circle(
        conn,
        channel_order,
        node_colors=node_colors,
        title=title,
        n_lines=300,
        colormap='Blues',
        #vmin=threshold,
        #vmax=np.max(conn),
        linewidth=2,
        fig=None,
        show=True
    )
    print('plot done!')

def prepare_mne_connectivity(mean_pvals, binary_adj, channel_order):
    # reorder
    mean_pvals = mean_pvals.loc[channel_order, channel_order]
    binary_adj = binary_adj.loc[channel_order, channel_order]

    # causal strength
    strength = -np.log10(mean_pvals.values + 1e-12)

    # mask non-significant
    strength[binary_adj.values == 0] = 0.0

    return strength