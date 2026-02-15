import numpy as np

#%% ---------- Granger ----------
def aggregate_granger(results, max_pval=0.05): # order: avg lags, avg subjects
    """
    Aggregate Granger-causality significance across lags and subjects, 
    obtaining 1 ECN per group.

    For each diagnostic group (AD, FTD, CN), this function computes a group-level
    connection strength matrix by:
        1) Thresholding p-values at `max_pval` for each lag and subject
           (accepted links = 1, rejected links = 0),
        2) Averaging accepted links across lags for each subject,
        3) Averaging subject-level strengths within each group.

    The resulting strength values represent the empirical strength-probability that a
    directed connection is significant across lags and subjects.

    Parameters
    ----------
    results : dict
        Nested dictionary indexed by lag -> group -> "pvals".
        Expected structure:
            results[lag][group]["pvals"][subject]
        where each entry is a pandas DataFrame of p-values with identical
        row and column channel labels.

    max_pval : float, optional (default=0.05)
        Significance threshold for accepting a directed connection.

    Returns
    -------
    strengths_groups : dict
        Dictionary with keys {"AD", "FTD", "CN"}.
        Each entry contains:
            "strength" : numpy.ndarray of shape (n_channels, n_channels)
                Group-level directed connection strengths in [0, 1].

    Notes
    -----
    - The number of lags is inferred from `results`, but adding +1 to the 
    average count, in order to account for instantaneous causality and not only
    lagged (comparable to LiNGAM!).
    """
    ch_names = results[1]["AD"]["pvals"][0].columns # remind indexes' names = columns' names
    n_ch = len(ch_names)
    n_lags = len(results)+1 # lags count (+1 to account istantaneous lack => plots comparable with lingam!)

    # create strengths: avg accepted pvals across lags and subjects
    strengths_groups = {
        "AD":  {"strength": np.zeros((n_ch, n_ch))},
        "FTD": {"strength": np.zeros((n_ch, n_ch))},
        "CN":  {"strength": np.zeros((n_ch, n_ch))}}

    # lags aggregation
    for group in strengths_groups.keys():
        n_subs = len(results[1][group]["pvals"])
        group_strength_sum = np.zeros((n_ch, n_ch)) # empirical strength: for mean group strength

        for subj in range(n_subs):
            # mean on lags 
            subj_strength_sum = np.zeros((n_ch, n_ch))
            
            for lag in results.keys():  # 1,...,maxlag
                pvals = results[lag][group]["pvals"][subj]

                # keep only links w/ higher confidence
                p_masked = (pvals <= max_pval).astype(int) # 1->accepted, 0->not accepted

                subj_strength_sum += p_masked
            
            # mean strength for this subject
            subj_strength_mean = subj_strength_sum / n_lags

            group_strength_sum += subj_strength_mean

        # mean for group
        strengths_groups[group]["strength"] = group_strength_sum / n_subs
    
    return strengths_groups

def aggregate_granger2(results, max_pval=0.05): # order: avg subjects, avg lags
    """
    Compute group-level Granger-causality significance by aggregating
    across subjects first and across lags afterward.

    For each diagnostic group (AD, FTD, CN), this function:
        1) Averages p-values across subjects for each lag,
        2) Applies a significance threshold to the group-mean p-values,
        3) Averages accepted connections across lags.

    The resulting matrix represents the proportion of lags for which
    a directed connection is significant at the group level.

    Parameters
    ----------
    results : dict
        Nested dictionary indexed by lag -> group -> "pvals".
        Expected structure:
            results[lag][group]["pvals"][subject]
        where each element is a pandas DataFrame of p-values with identical
        row and column channel labels.

    max_pval : float, optional (default=0.05)
        Significance threshold applied to group-mean p-values.

    Returns
    -------
    strengths_groups : dict
        Dictionary with keys {"AD", "FTD", "CN"}.
        Each entry contains:

            "strength" : numpy.ndarray of shape (n_channels, n_channels)
                Proportion of lags in which the connection is significant
                at the group level (values in [0, 1]).

    Notes
    -----
    - The number of lags is inferred from `results`, but adding +1 to the 
    average count, in order to account for instantaneous causality and not only
    lagged (comparable to LiNGAM!).
    """
    ch_names = results[1]["AD"]["pvals"][0].columns # remind indexes' names = columns' names
    n_ch = len(ch_names)
    n_lags = len(results)+1 # lags count (+1 to account istantaneous lack => plots comparable with lingam!)

    # create strengths: avg accepted pvals across lags and subjects
    strengths_groups = {
        "AD":  {"strength": np.zeros((n_ch, n_ch))},
        "FTD": {"strength": np.zeros((n_ch, n_ch))},
        "CN":  {"strength": np.zeros((n_ch, n_ch))}}

    # lags aggregation
    for group in strengths_groups.keys():
        group_strength_sum = np.zeros((n_ch, n_ch)) # empirical strength: for mean group strength

        for lag in results.keys():
            # mean on lags 
            lag_pvals_mean = np.mean(results[lag][group]["pvals"], axis=0)

            # keep only links w/ higher confidence
            p_masked = (lag_pvals_mean <= max_pval).astype(int) # 1->accepted, 0->not accepted

            group_strength_sum += p_masked

        # mean for group
        strengths_groups[group]["strength"] = group_strength_sum / n_lags
    
    return strengths_groups


#%% ---------- LiNGAM ----------
def aggregate_lingam(results, min_prob=0.95): # order: avg lags, avg subjects
    """
    Aggregate LiNGAM connection strengths and support probabilities
    across lags and subjects.

    For each diagnostic group (AD, FTD, CN), this function computes
    group-level directed connection metrics by:
        1) Thresholding: selecting connections whose bootstrap probability is
           greater than or equal to `min_prob`,
        2) Averaging masked connection strengths across lags for
           each subject,
        3) Averaging subject-level results within each group.

    Two group-level matrices are produced:
        - Mean connection strength across lags and subjects
        - Mean proportion of accepted probabilities across lags and subjects

    Parameters
    ----------
    results : dict
        Nested dictionary indexed by lag -> group -> {"strengths", "probs"}.
        Expected structure:
            results[lag][group]["strengths"][subject]
            results[lag][group]["probs"][subject]
        where each entry is a pandas DataFrame with identical row and
        column channel labels.

    min_prob : float
        Minimum bootstrap probability required to retain a connection.

    Returns
    -------
    res_groups : dict
        Dictionary with keys {"AD", "FTD", "CN"}.
        Each entry contains:
            "strength" : numpy.ndarray of shape (n_channels, n_channels)
                Group-level mean directed connection strength.

            "n_probs" : numpy.ndarray of shape (n_channels, n_channels)
                Group-level mean proportion of accepted probabilities
                (values in [0, 1]).

    Notes
    -----
    - Lag 0 is interpreted as instantaneous effects if present.
    """    
    ch_names = results[0]["AD"]["strengths"][0].columns # remind indexes' names = columns' names
    n_ch = len(ch_names)
    n_lags = len(results) # istantaneous+lags count
    
    # aggregate strengths and probs: avg across lags and subjects for each group
    res_groups = {
        "AD":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
        "FTD": {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
        "CN":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))}}

    # lags aggregation
    for group in res_groups.keys():
        n_subs = len(results[0][group]["strengths"])
        group_strength_sum = np.zeros((n_ch, n_ch)) # for mean group strength
        group_n_probs_sum = np.zeros((n_ch, n_ch)) # for mean group n_probs

        for subj in range(n_subs):
            # mean on lags 
            subj_strength_sum = np.full((n_ch, n_ch), 0.0)
            subj_n_probs_sum = np.full((n_ch, n_ch), 0.0)
            
            for lag in results.keys():  # 0,...,maxlag
                s = results[lag][group]["strengths"][subj]
                p = results[lag][group]["probs"][subj]

                mask = (p >= min_prob) # keep only strengths w/ higher probabilities
                s_masked = np.where(mask, s, 0.0)

                subj_strength_sum += s_masked
                subj_n_probs_sum += mask.astype(int) # 1->accepted, 0->not accepted
            
            # mean strength and accepted probs for this subject
            subj_strength_mean = subj_strength_sum / n_lags
            subj_n_probs_mean = subj_n_probs_sum / n_lags

            group_strength_sum += subj_strength_mean
            group_n_probs_sum += subj_n_probs_mean

        # mean for group
        res_groups[group]["strength"] = group_strength_sum / n_subs
        res_groups[group]["n_probs"] = group_n_probs_sum / n_subs
    
    return res_groups


def aggregate_lingam2(results, min_prob=0.95): # order: avg subjects, avg lags
    """
    Compute group-level LiNGAM connection metrics by aggregating
    across subjects first and across lags afterward.

    For each diagnostic group (AD, FTD, CN), this function:
        1) Averages connection strengths and bootstrap probabilities
           across subjects for each lag,
        2) Applies a probability threshold to the group-mean probabilities,
        3) Averages thresholded strengths across lags.

    This procedure estimates a population-level causal structure.

    Two group-level matrices are produced:
        - Mean connection strength across lags after probability thresholding
        - Proportion of lags in which the connection passes the threshold

    Parameters
    ----------
    results : dict
        Nested dictionary indexed by lag -> group -> {"strengths", "probs"}.
        Expected structure:
            results[lag][group]["strengths"][subject]
            results[lag][group]["probs"][subject]
        where each element is a pandas DataFrame with identical row and
        column channel labels.

    min_prob : float
        Minimum bootstrap probability required to retain a connection
        at the group level.

    Returns
    -------
    res_groups : dict
        Dictionary with keys {"AD", "FTD", "CN"}.
        Each entry contains:

            "strength" : numpy.ndarray of shape (n_channels, n_channels)
                Group-level mean directed connection strength.

            "n_probs" : numpy.ndarray of shape (n_channels, n_channels)
                Proportion of lags in which the connection satisfies
                the probability threshold (values in [0, 1]).

    Notes
    -----
    - Lag 0 is interpreted as instantaneous effects if present.
    """
    ch_names = results[0]["AD"]["strengths"][0].columns # remind indexes' names = columns' names
    n_ch = len(ch_names)
    n_lags = len(results) # istantaneous+lags count
    
    # aggregate strengths and probs: avg across lags and subjects for each group
    res_groups = {
        "AD":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
        "FTD": {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
        "CN":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))}}

    # lags aggregation
    for group in res_groups.keys():
        group_strength_sum = np.zeros((n_ch, n_ch)) # for mean group strength
        group_n_probs_sum = np.zeros((n_ch, n_ch)) # for mean group n_probs

        for lag in results.keys():  # 0,...,maxlag
            s_mean = np.mean(results[lag][group]["strengths"], axis=0)
            p_mean = np.mean(results[lag][group]["probs"], axis=0)

            mask = (p_mean >= min_prob) # keep only strengths w/ higher probabilities
            s_masked = np.where(mask, s_mean, 0.0)

            group_strength_sum += s_masked
            group_n_probs_sum += mask.astype(int) # 1->accepted, 0->not accepted
    
        # mean for group
        res_groups[group]["strength"] = group_strength_sum / n_lags
        res_groups[group]["n_probs"] = group_n_probs_sum / n_lags
    
    return res_groups