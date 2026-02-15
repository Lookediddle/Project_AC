import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
plt.ion() # plt.show() not blocking execution

#%% ref: https://phdinds-aim.github.io/time_series_handbook/04_GrangerCausality/04_GrangerCausality.html
def granger_ecn(epochs, channels, maxlag=4, current_subject=None, ic=False):
    """
    Compute Granger ECN for one subject by aggregating across epochs 
    (mean p-values are returned).

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
    channels : dict, channel numbers mapped to names (e.g. {0:"Fp1", 1:"Fp2", etc.})
    maxlag : int
    current_subject : dict, if not None, it means that the stationarity was already checked 
        during preprocessing. So stationarity check is skipped when not necessary.
        Default at None.
    ic : bool, if True the information criteria for the VAR model are computed to select the 
        number of lags. Plots are showed.
        Default at False.

    Returns
    -------
    mean_pvals : dict
        maps each lag (key) to its pd.DataFrame of causal pvals (value, shape:(n_channels, n_channels))
    """

    ch_names = [name for ch,name in channels.items()] # Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz

    all_pvals_df = {} # lags to pvals
    for k in range(1,maxlag+1): # 1,...,maxlag
        all_pvals_df[k] = [] 

    for e, epoch in enumerate(epochs):
        print(f"... [epoch {e}]", end='-->', flush=True)
        epoch_df = pd.DataFrame(epoch.T, columns=ch_names)
        
        # 1. make epoch stationary 
        print('... stationariety', end=' ', flush=True)
        if current_subject == None:
            epoch_df, n_diffs, _ = make_stationary(epoch_df) 
        
        else: # stationarity check was done in pre-processing
            curr_epoch_report = current_subject['epochs'][e]
            if curr_epoch_report['n_diffs'] > 0: # not stationary!!! differencing was applied in this epoch
                epoch_df, n_diffs, _ = make_stationary(epoch_df) 

        # 2. select maxlag from multivariate time series
        if ic:
            print('selecting maxlag...')
            select_p(epoch_df) # select the VAR (Vector AutoRegressive) model order p (i.e. maxlag) from plots' elbows

        # 3. apply Granger
        print('... ***computing Granger***', end=', ', flush=True)
        pvals = granger_causation_matrix(epoch_df, epoch_df.columns, maxlag) 

        for k in range(1, maxlag+1): # granger: only lagged
            all_pvals_df[k].append(pvals[k])

    # 4. aggregate across epochs (i.e. mean), for each lag
    mean_pvals = {}
    for lag,res in all_pvals_df.items():
        mean_pvals[lag] = sum(res) / len(res)

    return mean_pvals


# stationarity (Granger assumption)
def make_stationary(series_df, verbose=True):
    """
    Make all channels in an epoch stationary.
    Iteratively difference the series until they are all stationary
    according to both ADF and KPSS tests.

    Parameters
    -------
    series_df (pd DataFrame) : multivariate time series (n_samples, n_channels)

    Returns
    -------
    epoch_df (pd DataFrame) : multivariate time series (n_samples, n_channels)
    n_diffs (int) : number of times that differencing is applied
    diffed_channels (list) : list of channels that were not stationary
    """
    epoch_df = series_df.copy()

    stationary_channels = {} # ch : (True|False)
    n_diffs = 0
    diffed_channels = [] # track differenced channels

    while True:
        if verbose: print('adf test', end=',', flush=True)
        adf = adf_test(epoch_df)
        if verbose: print('kpss test', end=',', flush=True)
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
                if verbose: print(f"differencing channel {ch}", end=',', flush=True)
                epoch_df[ch] = epoch_df[ch] - epoch_df[ch].shift(1)
                epoch_df = epoch_df.dropna() # drop rows with NaN values
                
                stationary_channels[ch] = False
                n_diffs += 1
                diffed_channels.append(ch)
            else:
                stationary_channels[ch] = True
        
        if all(stationary_channels.values()): break # if all channels are stationary (True), break

    return epoch_df, n_diffs, np.unique(diffed_channels)


def adf_test(data_df):
    """
    Augmented Dickey-Fuller test.
    Null hp: time series is NOT stationary (i.e., a unit root is present).
    
    data_df (DataFrame): multivariate time series (n_samples, n_channels)
    """
    test_stat, p_val = [], []
    cv_1pct, cv_5pct, cv_10pct = [], [], [] # critical values
    for c in data_df.columns: 
        adf_res = adfuller(data_df[c].dropna(), maxlag=10) # modified maxlag
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
    
    data_df (DataFrame): multivariate time series (n_samples, n_channels)
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
def select_p(data_df):
    """
    Plot metrics to select the order p of the VAR model (i.e. number of lags),
    e.g. when the curves do an elbow.
    Metrics correspond to different multivariate information criteria (AIC, BIC, HQIC), and FPE.

    data_df (DataFrame) : time series data.
    """
    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(data_df) 
    p = np.arange(1,10) # modified maxlag
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
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
    lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
    plt.tight_layout()
    for a in ax:
        a.grid(True)
        a.set_xlabel('Lags', labelpad=0)
    print(lags_metrics_df.idxmin(axis=0))

# Granger 
def granger_causation_matrix(data, variables, maxlag, test = 'ssr_chi2test', verbose=False):    
    """
    Check Granger Causality of all possible combinations of the time series (pairwise).
    The rows are the response variables, columns are predictors. The values in the returned table 
    are the P-Values. P-Values lesser than the significance level (e.g. 0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, i.e. the 'X does not cause Y' can be rejected.

    Parameters
    -----------
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    maxlag (int) : maximum number of lags.

    Returns
    -----------
    df (DataFrame) : dict 
        maps each lag to its table of resulting p-values (columns: X, rows: Y).
    """
    df={}
    for lag in range(1, maxlag+1):
        df[lag] = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    
    #print(f"- channels: ", end=' ', flush=True)
    for c in variables: #df.columns:
        for r in variables: #df.index:
            #print(f"({c}->{r})", end=' ', flush=True) # print channels being analysed
            test_result = grangercausalitytests(data[[r, c]], maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            #min_p_value = np.min(p_values) # (Luca: do not aggregate lags now!)
            for lag in range(1, maxlag+1):
                df[lag].loc[r, c] = p_values[lag-1]

    return df
