import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from lingam import VARLiNGAM
import numpy as np
from .granger import make_stationary

# time series
def lingam_ecn(epochs, channels, maxlag=4, threshold=0.01):
    """
    Compute VAR-LiNGAM (time-series) ECN for one subject by aggregating across epochs.

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
        #print("stationarity")
        #epoch_df, _ = make_stationary(epoch_df)

        # 2. VAR-LiNGAM
        print("Fitting VAR-LiNGAM...")
        model = VARLiNGAM(lags=maxlag, criterion=None) # criterion=None: do not search best lags
        model.fit(epoch_df)

        # adjacency_matrices_: list [B0, B1, ..., Bp]
        B_mats = model.adjacency_matrices_

        # aggregate causal effects (instantaneous and lagged (k=1,...,maxlag))
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



# istantaneous, pure lingam
from lingam import ICALiNGAM, DirectLiNGAM
def lingam_ecn_no_lags(epochs, channels, threshold=0.01):
    """
    Compute LiNGAM (istantaneous) ECN for one subject by aggregating across epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
    channels : dict, channel numbers mapped to names
    threshold : float
        Threshold on absolute causal strength

    Returns
    -------
    mean_strength : pd.DataFrame (n_channels x n_channels)
        Mean absolute causal strengths
    binary_adj : pd.DataFrame (n_channels x n_channels)
        Binary ECN adjacency
    """

    ch_names = [name for _, name in channels.items()]
    all_B = []

    for e, epoch in enumerate(epochs):
        print(f"\n-- epoch: {e} --")

        # LiNGAM expects shape (n_samples, n_channels)
        epoch_df = pd.DataFrame(epoch.T, columns=ch_names)

        print("Fitting LiNGAM...")
        model = DirectLiNGAM() # DirectLiNGAM converges to right solution (!= ICALiNGAM to local optima!)
        model.fit(epoch_df)

        # adjacency matrix (B_ij = i → j)
        B = pd.DataFrame(
            model.adjacency_matrix_,
            index=ch_names,
            columns=ch_names
        )

        all_B.append(abs(B))

    # aggregate across epochs
    mean_strength = sum(all_B) / len(all_B)

    # binary ECN
    binary_adj = (mean_strength > threshold).astype(int)

    return mean_strength, binary_adj
