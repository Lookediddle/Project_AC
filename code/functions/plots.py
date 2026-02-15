import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# list of str: exact order of channels around the circle
channel_order = ["C3", "F4", "F3", "Fp2", "Fp1", "Pz", "Cz", "Fz", "T6", "T5", "T4", "T3", "F8", "F7", "O2", "O1", "P4", "P3", "C4"]

# dict: channel -> color
channel_colors = {
    "Fp1": "#CD3264", "Fp2": "#4EA760", "F3":  "#E6D241", "F4":  "#3C82B4",
    "C3":  "#DF8B53", "C4":  "#9049A2", "Pz":  "#262673", "P3": "#6AC9C9",
    "P4": "#D960C7", "O1":  "#B5CC5C", "O2":  "#E3C3C3", "F7": "#418181",
    "F8": "#D1BBE6", "T3":  "#997347", "T4":  "#E6DEAD", "T5":  "#732626",
    "T6":  "#A6E6BF", "Fz":  "#737333", "Cz":  "#E6BFA6"
}

def plot_ecn(strength, threshold=0.1, ax=None, title=None, figsize=(6, 6), widths=None):
    """
    ECN chord diagram.

    Parameters
    ----------
    strength : pd.DataFrame
        causal strengths. Rows = targets (Y), Columns = sources (X)        
    threshold : float
        Minimum causal strength
    ax : matplotlib Axes or None
        If None, creates a standalone figure.
    title : str or None
    figsize : tuple
        Used only if ax is None.
    widths : pd.DataFrame
        widths for causal links. Rows = targets (Y), Columns = sources (X)
        Default None. 

    Returns
    ----------
    ax : matplotlib axes used for the plot/subplot
    """

    # ---- reorder matrix ----
    strength = strength.loc[channel_order, channel_order]

    n = len(channel_order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # node positions on circle
    pos = {
        ch: np.array([np.cos(a), np.sin(a)])
        for ch, a in zip(channel_order, angles)
    }

    # ---- manage plot or subplots ----
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- draw nodes ----
    for ch in channel_order:
        x, y = pos[ch]
        ax.scatter(
            x, y,
            s=500, # not too large otherwise it will be cut out
            color=channel_colors[ch],
            edgecolors="black",
            zorder=8
        )
        ax.text(
            1.15 * x,
            1.15 * y,
            ch,
            ha="center",
            va="center",
            fontsize=11
        )

    # ---- draw chord diagram ----        
    for src in channel_order:        # columns = sources
        for tgt in channel_order:    # rows = targets
            if src == tgt: # skip self
                continue

            w = strength.loc[tgt, src]
            if w < threshold: # skip values under minimum strength
                continue

            p0 = pos[src]
            p2 = pos[tgt]

            if widths is not None: # use chord widths            
                link_w = widths.loc[tgt, src]
                stretched_w = 5*link_w # linear
                #stretched_w = (5 / (math.e - 1)) * (math.e**link_w - 1) # exponential: x = (5/(e-1)) * (e^x - 1)
                draw_chord_arrow(ax, p0, p2, color=channel_colors[src], width=stretched_w)
            else:
                draw_chord_arrow(ax, p0, p2, color=channel_colors[src])

    if title is not None:
        ax.set_title(title, fontsize=14, pad=20)
    
    if standalone:
        plt.show()

    return ax


def draw_chord_arrow(ax, p0, p2, color, width=1):
    """
    Draw a chord from p0 (source node) to p2 (target node).
    """
    p1 = np.array([0.0, 0.0]) # control point: center

    verts = [
        (p0[0], p0[1]),  # start
        (p1[0], p1[1]),  # control
        (p2[0], p2[1]),  # end
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,  # quadratic Bezier
        Path.CURVE3,
    ]

    path = Path(verts, codes)

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=width,
        alpha=0.7,
        zorder=2
    )
    ax.add_patch(patch)

