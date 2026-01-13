import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
plt.ion() # plt.show() not blocking execution

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
def granger_ecn(epochs, maxlag=7, alpha=0.05):
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
        print(f"\nepoch: {e}")
        # 1. make epoch stationary
        epoch_df = pd.DataFrame(epoch.T)
        epoch_df, n_diffs = make_stationary(epoch_df) # *****da scommentare*****

        # 2. VAR model (Vector AutoRegressive) ************************LASCIARE SOLO SELECT(P) E COMMENTARE TUTTO IL RESTO? NON SERVE A UN CAZZO IL VAR SU TRAIN E TEST???**********
        #train_df, test_df = splitter(epoch_df) # split the data into train and test sets
        print('selecting maxlag...')
        select_p(epoch_df) #train_df) # select the VAR order p by computing the different multivariate information criteria (AIC, BIC, HQIC), and FPE
        p = maxlag
        #model = VAR(train_df)
        #var_model = model.fit(p) # fit the VAR model with the chosen order

        # 3. Apply Granger
        print('Computing Granger causation matrix...')
        pvals = granger_causation_matrix(epoch_df, epoch_df.columns, p) #train_df, train_df.columns, p)

        all_pvals[e] = pvals

    # 4. Aggregate across epochs
    mean_pvals = np.mean(all_pvals, axis=0)

    # 5. ECN adjacency (binary)
    binary_adj = (mean_pvals < alpha).astype(int)

    return mean_pvals, binary_adj


# stazionarietà (Granger assumption)
def make_stationary(series_df):
    """
    Make all channels in an epoch stationary.
    Iteratively difference the series until they are all stationary
    according to both ADF and KPSS tests.

    Returns
    -------
    epoch_df (pd dataframe) : time series (n_channels, n_samples)
    n_diffs (int) : number of times that differencing is applied
    """
    epoch_df = series_df.copy()

    stationary_channels = {} # ch : (True|False)
    n_diffs = 0

    while True:
        print('adf test...')
        adf = adf_test(epoch_df)
        print('kpss test...')
        kpss = kpss_test(epoch_df)

        # Series is stationary if ADF rejects unit root and KPSS does NOT reject stationarity
        for ch in epoch_df.columns:
            # check adf
            if (abs(adf[ch]['Test statistic']) > abs(adf[ch]['Critical value - 1%'])
                and abs(adf[ch]['Test statistic']) > abs(adf[ch]['Critical value - 5%'])
                and abs(adf[ch]['Test statistic']) > abs(adf[ch]['Critical value - 10%'])):
                adf_stationarity = True # reject null hp -> series is stationary :)
            else:
                adf_stationarity = False
            
            # check kpss
            if (abs(kpss[ch]['Test statistic']) > abs(kpss[ch]['Critical value - 1%'])
                and abs(kpss[ch]['Test statistic']) > abs(kpss[ch]['Critical value - 5%'])
                and abs(kpss[ch]['Test statistic']) > abs(kpss[ch]['Critical value - 10%'])):
                kpss_stationarity = False # reject null hp -> series is NOT stationary :(
            else:
                kpss_stationarity = True

            if adf_stationarity==False or kpss_stationarity==False:
                # apply differencing on this channel
                print(f"differencing channel {ch}")
                epoch_df[ch] = epoch_df[ch] - epoch_df[ch].shift(1)
                epoch_df = epoch_df.dropna() # drop rows with NaN values
                
                stationary_channels[ch] = False
                n_diffs += 1
            else:
                stationary_channels[ch] = True
        
        if all(stationary_channels.values()): break # if all channels are stationary (True), break

    return epoch_df, n_diffs


def adf_test(data_df):
    """
    Augmented Dickey-Fuller test.
    Null hp: time series is NOT stationary (i.e., a unit root is present).
    
    :param data_df: all channels time series
    """
    test_stat, p_val = [], []
    cv_1pct, cv_5pct, cv_10pct = [], [], [] # critical values
    for c in data_df.columns: 
        adf_res = adfuller(data_df[c].dropna(), maxlag=20)
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


def kpss_test(data_df):
    """
    Kwiatkowski-Phillips-Schmidt-Shin test.
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


# VAR model processing
def splitter(data_df):
    """
    Split data into train and test, 80% and 20% respectively. 
    
    :param data_df: time series to split.
    """
    end = round(len(data_df)*.8)
    train_df = data_df[:end]
    test_df = data_df[end:]
    return train_df, test_df

def select_p(train_df):
    """
    Show some metrics to select the order of the VAR model (i.e. number of lags)
    
    :param train_df: time series data.
    """
    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(train_df) 
    p = np.arange(1,20)
    for i in p:
        result = model.fit(i)
        aic.append(result.aic)
        bic.append(result.bic)
        fpe.append(result.fpe)
        hqic.append(result.hqic)
    lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                    'BIC': bic, 
                                    'HQIC': hqic,
                                    'FPE': fpe}, 
                                   index=p)    
    fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharex=True)
    lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
    plt.tight_layout()
    for a in ax:
        a.grid(True)
    print(lags_metrics_df.idxmin(axis=0))

# Granger 
def granger_causation_matrix(data, variables, p, test = 'ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the time series.
    The rows are the response variables, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    p : maxlag (int).
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    
    print(f"- channels: ", end=' ', flush=True)
    for c in df.columns:
        for r in df.index:
            print(f"({c}->{r})", end=' ', flush=True)
            test_result = grangercausalitytests(data[[r, c]], p, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(p)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    print(' ')
    return df