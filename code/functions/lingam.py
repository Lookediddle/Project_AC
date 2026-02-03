import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from lingam import VARLiNGAM
import numpy as np
from .granger import make_stationary
from statsmodels.tsa.api import VAR
from scipy.stats import chi2 # for Wald test on VAR coefs

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
    mean_strength : dict
        maps each lag to its pd.DataFrame of causal strengths (n_channels, n_channels)
    """

    ch_names = [name for _, name in channels.items()]
    n_channels = len(ch_names)
    strength_df = {} # {lag:strength_matrix}, e.g.: {0:B0, ..., maxlag:Bmaxlag}
    all_pvals_df = {} # lags to pvals, test on VAR coefficients
    for k in range(maxlag+1):
        strength_df[k] = [] # init
        if k != 0: # not istantaneous! only for VAR coefficients (i.e. lags)
            all_pvals_df[k] = [] 

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

        B_mats = model.adjacency_matrices_ # adjacency_matrices_: list [B0, B1, ..., Bp]

        # if verbose: print("... independence pvalues ...", end=', ', flush=True)
        # pvals_independence = model.get_error_independence_p_values() #****solo su B0 istantanea

        for k in range(len(B_mats)): # k=0,...,maxlag
            strength = pd.DataFrame(B_mats[k], index=ch_names, columns=ch_names)
            strength_df[k].append(strength)

        # 3. confidence: test on VAR coefficients (lagged effects only)
        if verbose: print('...test VAR params...', end=' ', flush=True)
        var_model = VAR(epoch_df)
        var_res = var_model.fit(maxlag, trend='n') # == VAR in VAR-LiNGAM

        for k in range(1, maxlag+1): # not 0, only lagged
            pvals = pd.DataFrame(
                np.ones((n_channels, n_channels)),
                index=ch_names,
                columns=ch_names
            )

            for ch_to in ch_names:
                for ch_frm in ch_names:
                    if ch_to == ch_frm: 
                        continue

                    _, pvals.loc[ch_to, ch_frm] = wald_test_var_coef(var_res, ch_to, ch_frm, k)

            all_pvals_df[k].append(pvals)


        # pvals = var_res.pvalues # p-values for model coefficients from Student t-distribution
        # for lag in range(1, maxlag+1):
        #     pval_df = pd.DataFrame( # init
        #         np.ones((len(ch_names), len(ch_names))),
        #         index=ch_names,
        #         columns=ch_names
        #     )
        #     for to in ch_names:
        #         for frm in ch_names:
        #             if to == frm:
        #                 continue

        #             param = f"L{lag}.{frm}"
        #             if param in pvals.index:
        #                 pval_df.loc[to, frm] = pvals.loc[param, to]
            
        #     all_pvals_df[lag].append(pval_df)


    # aggregate across epochs (i.e. mean), for each lag
    mean_strength = {}
    for lag,res in strength_df.items():
        mean_strength[lag] = sum(res) / len(res)
    
    mean_pvals = {}
    for lag,res in all_pvals_df.items():
        mean_pvals[lag] = sum(res) / len(res)

    return mean_strength, mean_pvals


# Wald test on VAR coefficients
def wald_test_var_coef(var_res, ch_to, ch_frm, k):
    """
    Wald test for H0: A_k(ch_to,ch_frm) = 0 (i.e. no causal strength),
    where A_k(ch_to,ch_frm) is a VAR coefficient at lag k.

    Returns
    -------
    W : float
        Wald statistic
    p_value : float
    """

    param = f"L{k}.{ch_frm}"
    if param not in var_res.params.index:
        raise ValueError(f"Coefficient label {param} doesn't exist.")

    # coefficient estimate
    theta_hat = var_res.params.loc[param, ch_to]

    # variance estimate
    cov = var_res.cov_params()
    var_theta = cov.loc[(param, ch_to), (param, ch_to)]

    # Wald
    W = (theta_hat ** 2) / var_theta
    p_value = 1 - chi2.cdf(W, df=1)

    return W, p_value


# varlingam + bootstrap
def lingam_ecn_boot(epochs, channels, maxlag=4, current_subject=None, verbose=True):
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
        print("... bootstrapping ...", end=', ', flush=True)
        boot_res = model.bootstrap(epoch_df, n_sampling=10)
        print("... getting probabilities", end=', ', flush=True)
        probs = boot_res.get_probabilities()


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
