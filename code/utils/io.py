import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import re

def save_results(
    data_dict=None,
    base_dir="results",
    dpi=300,
    close_figures=False,
    save_pickle=True
):
    """
    Save all open figures and associated data inside a timestamped folder.

    Parameters
    ----------
    data_dict : dict
        Dictionary of data to save (e.g. arrays, scalars, lists)
    base_dir : str
        Root results directory
    dpi : int
        Resolution for saved figures
    close_figures : bool
        Close figures after saving
    save_pickle : bool
        Save data as pickle
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(base_dir, timestamp)

    fig_dir = os.path.join(root_dir, "figures")
    data_dir = os.path.join(root_dir, "data")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save figures
    fig_nums = plt.get_fignums()

    for i, fig_num in enumerate(fig_nums, start=1):
        fig = plt.figure(fig_num)

        title = ""
        if fig._suptitle is not None:
            title = fig._suptitle.get_text()
        elif fig.axes:
            title = fig.axes[0].get_title()

        safe_title = _sanitize_filename(title)
        filename = f"{i}_{safe_title}.png"
        path = os.path.join(fig_dir, filename)

        #fig.tight_layout()
        fig.savefig(path, dpi=dpi)

        print(f"\n[Saved figure] {path}")

        if close_figures:
            plt.close(fig)

    # Save data
    if data_dict is not None:
        if save_pickle:
            pkl_path = os.path.join(data_dir, "saved_data.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data_dict, f)
            print(f"\n[Saved data] {pkl_path}")

    return root_dir



def _sanitize_filename(text, max_len=50):
    """
    Make a string safe to use as a filename.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)          # spaces -> _
    text = re.sub(r"[^a-z0-9_]", "", text)    # remove invalid chars
    return text[:max_len] if text else "figure"



def load_data(path):
    """
    Load a pickle file containing a dictionary of saved variables.

    Parameters
    ----------
    path : str
        Path to the pickle file (e.g. results/20260114_153210/data/saved_data.pkl)

    Returns
    -------
    data : dict
        Saved dictionary dictionary
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in pickle file, got {type(data)} instead.")
    
    return data 
