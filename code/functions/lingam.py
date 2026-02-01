import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from lingam import VARLiNGAM
import numpy as np
from .granger import make_stationary

# time series
def lingam_ecn(epochs, channels, maxlag=4, current_subject=None, verbose=True):
    """
    Compute VAR-LiNGAM (time-series) ECN for one subject by aggregating across epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
    channels : dict, channel numbers mapped to names (e.g. {0:"Fp1", 1:"Fp2", etc.})
    maxlag : int
    alpha : float
        Confidence level used to check independence (i.e. LiNGAM assumption)
    current_subject : dict, if not None, it means that the stationarity was already checked 
        during preprocessing. So stationarity check is skipped when not necessary.
        Default at None.

    Returns
    -------
    mean_strength : pd.DataFrame (n_channels, n_channels)
    """

    ch_names = [name for _, name in channels.items()]
    all_strengths = []

    for e, epoch in enumerate(epochs):
        if verbose: print(f"... [epoch {e}]", end='-->', flush=True)

        epoch_df = pd.DataFrame(epoch.T, columns=ch_names)

        # 1. make epoch stationary (for VAR model!)
        if verbose: print('... stationariety', end=' ', flush=True)
        if current_subject == None:
            epoch_df, n_diffs, _ = make_stationary(epoch_df, verbose) 
        
        else: # stationarity check was done in pre-processing
            curr_epoch_report = current_subject['epochs'][e]
            if curr_epoch_report['n_diffs'] > 0: # not stationary!!! differencing was applied in this epoch
                epoch_df, n_diffs, _ = make_stationary(epoch_df, verbose) 

        # 2. apply VAR-LiNGAM
        if verbose: print("... ***fitting VAR-LiNGAM***", end=', ', flush=True)
        model = VARLiNGAM(lags=maxlag, criterion=None) # criterion=None: do not search best lags
        model.fit(epoch_df)

        # reliability: bootstrapping*****lentissimo
        # print("... bootstrapping ...", end=', ', flush=True)
        # boot_res = model.bootstrap(epoch_df, n_sampling=10)
        # print("... getting probabilities", end=', ', flush=True)
        # probs = boot_res.get_probabilities()


        # adjacency_matrices_: list [B0, B1, ..., Bp]
        B_mats = model.adjacency_matrices_

        lagged_strength = np.zeros_like(B_mats[0]) # aggregate causal effects (instantaneous and lagged (k=1,...,maxlag))

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

    #binary_adj = (mean_strength > threshold).astype(int) # *****ha senso tenerlo così?**

    return mean_strength#, binary_adj


# jackknife for reliability -> leave-one-epoch-out
def lingam_ecn_jk(epochs, channels, maxlag=4, current_subject=None, verbose=False):
    
    ch_names = [name for _, name in channels.items()]

    # jackknife
    N = len(epochs)
    jk_strengths = []

    print('\n... jackknife w/o epoch:', end=' ', flush=True)
    for i in range(N):
        print(f'{i}', end=',', flush=True)
        epochs_jk = np.delete(epochs, i, axis=0)
        mean_jk = lingam_ecn(
            epochs_jk, channels, maxlag, current_subject, verbose
        )
        jk_strengths.append(mean_jk.values)
    
    jk_strengths = np.array(jk_strengths)

    mean_jk = np.mean(jk_strengths, axis=0)

    # jackknife variance: <variance => >link stability
    var_jk = ((N-1)/N) * np.sum((jk_strengths - mean_jk)**2, axis=0)

    std_jk = np.sqrt(var_jk)

    return std_jk


# istantaneous, pure lingam******
# from lingam import ICALiNGAM, DirectLiNGAM
# def lingam_ecn_no_lags(epochs, channels, threshold=0.01):
#     """
#     Compute LiNGAM (istantaneous) ECN for one subject by aggregating across epochs.

#     Parameters
#     ----------
#     epochs : ndarray, shape (n_epochs, n_channels, n_samples)
#     channels : dict, channel numbers mapped to names
#     threshold : float
#         Threshold on absolute causal strength

#     Returns
#     -------
#     mean_strength : pd.DataFrame (n_channels x n_channels)
#         Mean absolute causal strengths
#     binary_adj : pd.DataFrame (n_channels x n_channels)
#         Binary ECN adjacency
#     """

#     ch_names = [name for _, name in channels.items()]
#     all_B = []

#     for e, epoch in enumerate(epochs):
#         print(f"... [epoch {e}]", end='-->', flush=True)

#         # LiNGAM expects shape (n_samples, n_channels)
#         epoch_df = pd.DataFrame(epoch.T, columns=ch_names)

#         print("... ***fitting LiNGAM*** ...", end=', ', flush=True)
#         model = DirectLiNGAM() # DirectLiNGAM converges to right solution (!= ICALiNGAM to local optima!)
#         model.fit(epoch_df)

#         # adjacency matrix (B_ij = i → j)
#         B = pd.DataFrame(
#             model.adjacency_matrix_,
#             index=ch_names,
#             columns=ch_names
#         )

#         all_B.append(abs(B))

#     # aggregate across epochs (i.e. mean)
#     mean_strength = sum(all_B) / len(all_B)

#     # ECN adjacency (binary) ***secondo me non serve, piuttosto i pvals (se calcolabili)***
#     binary_adj = (mean_strength > threshold).astype(int)

#     return mean_strength, binary_adj
