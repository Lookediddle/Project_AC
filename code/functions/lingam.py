import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from lingam import VARLiNGAM
import numpy as np
from .granger import make_stationary

def lingam_ecn(epochs, channels, maxlag=6, threshold=0.01):
    """
    Compute VAR-LiNGAM ECN for one subject by aggregating across epochs.

    Returns
    -------
    mean_strength : pd.DataFrame (n_channels, n_channels)
    binary_adj : pd.DataFrame (n_channels, n_channels)
    """

    ch_names = [name for _, name in channels.items()]
    all_strengths = []

    for e, epoch in enumerate(epochs):
        print(f"\n-- epoch: {e} --")

        epoch_df = pd.DataFrame(epoch.T, columns=ch_names)

        # 1. stationarity (same as Granger)
        print("stationarity")
        epoch_df, _ = make_stationary(epoch_df)

        # 2. VAR-LiNGAM
        print("Fitting VAR-LiNGAM...")
        model = VARLiNGAM()#*************************lags=maxlag, criterion=None) # criterion=None: do not search best lags
        model.fit(epoch_df)

        # adjacency_matrices_: list [B0, B1, ..., Bp]
        B_mats = model.adjacency_matrices_

        # aggregate causal effects (lagged and instantaneous)
        lagged_strength = np.zeros_like(B_mats[0])

        for k in range(len(B_mats)): # k=0,...,maxlag
            lagged_strength = np.maximum(lagged_strength, np.abs(B_mats[k]))


        strength_df = pd.DataFrame(
            lagged_strength,
            index=ch_names,
            columns=ch_names
        )

        all_strengths.append(strength_df)

    # aggregate across epochs (i.e. mean)
    mean_strength = sum(all_strengths) / len(all_strengths)

    binary_adj = (mean_strength > threshold).astype(int) # *****ha senso tenerlo così?**

    return mean_strength, binary_adj
