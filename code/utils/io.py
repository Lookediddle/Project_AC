import os
from datetime import datetime
import matplotlib.pyplot as plt


def save_all_open_figures(
    subdir="figures",
    base_dir="results",
    dpi=300,
    close=False
):
    """
    Save all currently open matplotlib figures.

    Parameters
    ----------
    subdir : str
        Subdirectory inside results/
    base_dir : str
        Root results directory
    dpi : int
        Resolution for saved images
    close : bool
        Close figures after saving
    """

    fig_nums = plt.get_fignums()

    if not fig_nums:
        print("[Info] No open figures to save.")
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []

    for i, fig_num in enumerate(fig_nums, start=1):
        fig = plt.figure(fig_num)

        # 1. Try suptitle
        title = ""
        if fig._suptitle is not None:
            title = fig._suptitle.get_text()

        # 2. Fallback: first axis title
        if not title and fig.axes:
            title = fig.axes[0].get_title()

        safe_title = _sanitize_filename(title)
        filename = f"{i}_{safe_title}_{timestamp}.png"
        path = os.path.join(out_dir, filename)

        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        saved_paths.append(path)

        print(f"[Saved figure] {path}")

        if close:
            plt.close(fig)

    return saved_paths


import re


def _sanitize_filename(text, max_len=50):
    """
    Make a string safe to use as a filename.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)          # spaces -> _
    text = re.sub(r"[^a-z0-9_]", "", text)    # remove invalid chars
    return text[:max_len] if text else "figure"

