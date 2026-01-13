import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

#%% https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html
# def aggregate_granger(epochs, maxlag=4, alpha=0.05):
#     """
#     Compute aggregated Granger adjacency for all epochs.

#     Returns
#     -------
#     mean_pvals : ndarray, shape (n_channels, n_channels)
#     binary_adj : ndarray, shape (n_channels, n_channels)
#     """

#     n_epochs, n_channels, _ = epochs.shape
#     all_pvals = np.zeros((n_epochs, n_channels, n_channels))

#     for idx in range(n_epochs):
#         print('epoch: ', idx)
#         all_pvals[idx] = granger_ecn_epoch(epochs[idx], maxlag=maxlag, alpha=alpha)
#         binary_adj = (all_pvals[idx] < alpha).astype(int) # ************************************************

#     mean_pvals = np.mean(all_pvals, axis=0)

#     # binary adjacency: 1 if mean p < significance level
#     binary_adj = (mean_pvals < alpha).astype(int)

#     return mean_pvals, binary_adj


# def granger_ecn_epoch(epoch, maxlag=10, alpha=0.05):
#     """
#     Compute Granger causality adjacency matrix for an EEG epoch.

#     Parameters
#     ----------
#     epoch : ndarray, shape (n_channels, n_samples_epoch)
#     maxlag : int
#     alpha : float
#         Significance level

#     Returns
#     -------
#     adj_matrix : ndarray, shape (n_channels, n_channels)
#         p-values (or binary adjacency)
#     """

#     n_channels, _ = epoch.shape
#     adj_matrix = np.ones((n_channels, n_channels))

#     print(f"- channels: ", end=' ', flush=True)
#     for i in range(n_channels):
#         for j in range(n_channels):
            
#             if not i == j:
#                 print(f"({i}, {j})", end=' ', flush=True)    
#                 p_val, f_stat = granger_pairwise(
#                     epoch[i], epoch[j], maxlag=maxlag
#                 )
#                 # p-values come from best lag
#                 adj_matrix[i, j] = p_val

#     print(' ')

#     return adj_matrix


# def granger_pairwise(x, y, maxlag=10, verbose=False):
#     """
#     Test if x Granger-causes y.

#     Parameters
#     ----------
#     x, y : 1D arrays
#         Time series signals of the same length
#     maxlag : int
#         Maximum lag to test
#     verbose : bool
#         Whether to print statsmodels output

#     Returns
#     -------
#     best_pvalue : float
#         Best (smallest) p-value among lags 1..maxlag
#     best_f_stat : float
#         Corresponding F statistic
#     """

#     # concatenate per statsmodels API
#     data = np.column_stack([y, x])
#     results = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

#     # extract best p-value across tested lags
#     best_pvalue = np.inf
#     best_f_stat = 0

#     for lag, test_res in results.items():
#         f_stat = test_res[0]["ssr_ftest"][0]
#         p_value = test_res[0]["ssr_ftest"][1]

#         if p_value < best_pvalue:
#             best_pvalue = p_value
#             best_f_stat = f_stat

#     return best_pvalue, best_f_stat


#%% https://phdinds-aim.github.io/time_series_handbook/04_GrangerCausality/04_GrangerCausality.html
def granger_ecn(epochs, maxlag=4, alpha=0.05):
    """
    Compute Granger ECN for one subject by aggregating across epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
    maxlag : int
    alpha : float

    Returns
    -------
    mean_pvals : ndarray (n_channels, n_channels)
    binary_adj : ndarray (n_channels, n_channels)
    """

    n_epochs, n_channels, _ = epochs.shape
    all_pvals = np.zeros((n_epochs, n_channels, n_channels))

    for e, epoch in enumerate(epochs):
        # 1. Make epoch stationary
        stat_epoch = make_epoch_stationary(epoch, alpha=alpha)

        # 2. Apply Granger VAR
        pvals = granger_var_epoch(
            stat_epoch,
            maxlag=maxlag,
            alpha=alpha
        )

        all_pvals[e] = pvals

    # 3. Aggregate across epochs
    mean_pvals = np.mean(all_pvals, axis=0)

    # 4. ECN adjacency (binary)
    binary_adj = (mean_pvals < alpha).astype(int)

    return mean_pvals, binary_adj


def granger_var_epoch(stat_epoch, maxlag=4, alpha=0.05):
    """
    Compute Granger causality adjacency matrix for one stationary epoch
    using a multivariate VAR model.

    Parameters
    ----------
    stat_epoch : ndarray, shape (n_channels, n_samples)
        Stationary EEG epoch
    maxlag : int
        Fixed VAR order
    alpha : float
        Significance level

    Returns
    -------
    pval_matrix : ndarray, shape (n_channels, n_channels)
        Granger causality p-values (i -> j)
    """

    n_channels, _ = stat_epoch.shape

    # VAR expects (time, variables)
    df = pd.DataFrame(stat_epoch.T)

    model = VAR(df)
    results = model.fit(maxlags=maxlag, ic=None)

    pval_matrix = np.ones((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue

            # Test: does channel i Granger-cause channel j?
            test = results.test_causality(
                caused=j,
                causing=[i],
                kind="f"
            )
            pval_matrix[i, j] = test.pvalue

    return pval_matrix


def make_epoch_stationary(epoch, alpha=0.05, max_diff=2):
    """
    Make all channels in an epoch stationary.
    Input:
      epoch : ndarray, shape (n_channels, n_samples)
      alpha: confidence interval
      max_diff: max number of differentiation
    Returns:
      station_epoch : ndarray, shape (n_channels, <reduced_samples>)
    """
    n_channels, _ = epoch.shape
    processed = []
    diffs_list = []

    epoch_df = pd.DataFrame(epoch.T)

    stat = make_stationary(epoch_df, alpha=alpha, max_diff=max_diff)
    processed.append(stat)

    # Find minimum length after differencing
    min_len = min([len(x) for x in processed])
    processed_truncated = np.array(
        [x[-min_len :] for x in processed]
    )

    return processed_truncated


def make_stationary(series_df, max_diff=2, alpha=0.05):
    """
    Iteratively difference the series until they are all stationary
    according to both ADF and KPSS tests.

    Returns
    -------
    epoch_df (pd dataframe) : 
    """
    epoch_df = series_df.copy()

    stationary_channels = {} # ch : (True|False)

    while True:
        print('adf test...')
        adf = adf_test(epoch_df)#, alpha=alpha)
        print('kpss test...')
        kpss = kpss_test(epoch_df)#, alpha=alpha)

        # Series is stationary if ADF rejects unit root and KPSS does NOT reject stationarity
        for ch in epoch_df.columns:
            # check adf
            if (abs(adf[ch]['Test statistic']) > abs(adf[ch]['Critical value - 1%'])):
                adf_stationarity = True # reject null hp -> series is stationary :)
            else:
                adf_stationarity = False
            
            # check kpss
            if (abs(kpss[ch]['Test statistic']) > abs(kpss[ch]['Critical value - 1%'])):
                kpss_stationarity = False # reject null hp -> series is NOT stationary :(
            else:
                kpss_stationarity = True

            if adf_stationarity==False or kpss_stationarity==False:
                # apply differencing on this channel
                print(f"differencing channel {ch}")
                epoch_df[ch] = epoch_df[ch] - epoch_df[ch].shift(1)
                epoch_df = epoch_df.dropna() # drop rows with NaN values
                
                stationary_channels[ch] = False
            else:
                stationary_channels[ch] = True
        
        if all(stationary_channels.values()): break # if all channels are stationary (True), break

    return epoch_df


def adf_test(data_df):
    """
    Augmented Dickey-Fuller test.
    Null hp: time series is NOT stationary (i.e., a unit root is present).
    
    :param data_df: all channels time series
    """
    test_stat, p_val = [], []
    cv_1pct, cv_5pct, cv_10pct = [], [], [] # critical values
    for c in data_df.columns: 
        adf_res = adfuller(data_df[c].dropna())
        test_stat.append(adf_res[0]) # always negative
        p_val.append(adf_res[1])
        cv_1pct.append(adf_res[4]['1%']) # always negative
        cv_5pct.append(adf_res[4]['5%'])
        cv_10pct.append(adf_res[4]['10%'])
    adf_res_df = pd.DataFrame({'Test statistic': test_stat, 
                               'p-value': p_val, 
                               'Critical value - 1%': cv_1pct,
                               'Critical value - 5%': cv_5pct,
                               'Critical value - 10%': cv_10pct}, 
                             index=data_df.columns).T
    adf_res_df = adf_res_df.round(4)
    return adf_res_df

# def adf_test(series, alpha=0.05):
#     """
#     Augmented Dickey-Fuller test.
#     Returns True if series is stationary (i.e., ADF rejects unit root).
#     """
#     result = adfuller(series, autolag="AIC")
#     pvalue = result[1]
#     return pvalue < alpha, pvalue

def kpss_test(data_df):
    """
    KPSS test.
    Null hp: time series is stationary.
    
    :param data_df: all channels time series
    """
    test_stat, p_val = [], []
    cv_1pct, cv_2p5pct, cv_5pct, cv_10pct = [], [], [], [] # critical values
    for c in data_df.columns: 
        kpss_res = kpss(data_df[c].dropna(), regression='ct')
        test_stat.append(kpss_res[0]) # always positive
        p_val.append(kpss_res[1])
        cv_1pct.append(kpss_res[3]['1%']) # always positive
        cv_2p5pct.append(kpss_res[3]['2.5%'])
        cv_5pct.append(kpss_res[3]['5%'])
        cv_10pct.append(kpss_res[3]['10%'])
    kpss_res_df = pd.DataFrame({'Test statistic': test_stat, 
                               'p-value': p_val, 
                               'Critical value - 1%': cv_1pct,
                               'Critical value - 2.5%': cv_2p5pct,
                               'Critical value - 5%': cv_5pct,
                               'Critical value - 10%': cv_10pct}, 
                             index=data_df.columns).T
    kpss_res_df = kpss_res_df.round(4)
    return kpss_res_df

# def kpss_test(series, alpha=0.05, regression="ct"):
    # """
    # KPSS test.
    # Returns True if series is stationary (i.e., KPSS DOES NOT reject stationarity).
    # """
#     statistic, pvalue, _, _ = kpss(series, regression=regression, nlags="auto")
#     return pvalue > alpha, pvalue
