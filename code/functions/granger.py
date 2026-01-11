import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

def aggregate_granger(epochs, maxlag=4, alpha=0.05):
    """
    Compute aggregated Granger adjacency for all epochs.

    Returns
    -------
    mean_pvals : ndarray, shape (n_channels, n_channels)
    binary_adj : ndarray, shape (n_channels, n_channels)
    """

    n_epochs, n_channels, _ = epochs.shape
    all_pvals = np.zeros((n_epochs, n_channels, n_channels))

    for idx in range(n_epochs):
        print('epoch: ', idx)
        all_pvals[idx] = granger_ecn_epoch(epochs[idx], maxlag=maxlag, alpha=alpha)
        binary_adj = (all_pvals[idx] < alpha).astype(int) # ************************************************

    mean_pvals = np.mean(all_pvals, axis=0)

    # binary adjacency: 1 if mean p < significance level
    binary_adj = (mean_pvals < alpha).astype(int)

    return mean_pvals, binary_adj


def granger_ecn_epoch(epoch, maxlag=10, alpha=0.05):
    """
    Compute Granger causality adjacency matrix for an EEG epoch.

    Parameters
    ----------
    epoch : ndarray, shape (n_channels, n_samples_epoch)
    maxlag : int
    alpha : float
        Significance level

    Returns
    -------
    adj_matrix : ndarray, shape (n_channels, n_channels)
        p-values (or binary adjacency)
    """

    n_channels, _ = epoch.shape
    adj_matrix = np.ones((n_channels, n_channels))

    print(f"- channels: ", end=' ', flush=True)
    for i in range(n_channels):
        for j in range(n_channels):
            
            if not i == j:
                print(f"({i}, {j})", end=' ', flush=True)    
                p_val, f_stat = granger_pairwise(
                    epoch[i], epoch[j], maxlag=maxlag
                )
                # p-values come from best lag
                adj_matrix[i, j] = p_val

    print(' ')

    return adj_matrix


def granger_pairwise(x, y, maxlag=10, verbose=False):
    """
    Test if x Granger-causes y.

    Parameters
    ----------
    x, y : 1D arrays
        Time series signals of the same length
    maxlag : int
        Maximum lag to test
    verbose : bool
        Whether to print statsmodels output

    Returns
    -------
    best_pvalue : float
        Best (smallest) p-value among lags 1..maxlag
    best_f_stat : float
        Corresponding F statistic
    """

    # concatenate per statsmodels API
    data = np.column_stack([y, x])
    results = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

    # extract best p-value across tested lags
    best_pvalue = np.inf
    best_f_stat = 0

    for lag, test_res in results.items():
        f_stat = test_res[0]["ssr_ftest"][0]
        p_value = test_res[0]["ssr_ftest"][1]

        if p_value < best_pvalue:
            best_pvalue = p_value
            best_f_stat = f_stat

    return best_pvalue, best_f_stat

