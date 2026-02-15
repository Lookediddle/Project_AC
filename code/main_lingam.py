#%% imports
from functions import *
from utils.io import *
from pathlib import Path

#%% hyperparams
fs_res = 50 # resampling frequency (Hz)
n_epochs = 10
maxlag = 4
alpha = 0.05

#%% dataset
dataset_dir = Path("../dataset/derivatives/")
subjects = sorted([p for p in dataset_dir.iterdir() if p.name.startswith("sub-")])
# groups reference
ranges = [(range(1, 37), "AD"), (range(37, 66), "CN"),(range(66, 89), "FTD")]
subs_to_groups = {num:label for r,label in ranges for num in r} # e.g. {1:"AD", 2:"AD", ...}

#%% VAR(-lingam) preprocessing: check stationarity
# print('--- check stationarity ---')
# all_subs_report = {}
# for subj_dir in subjects:
#     print(f"\n- {subj_dir.name} -", end=' ', flush=True)
#     report = analyze_subject_stationarity(subj_dir, resample=fs_res, n_epochs=n_epochs)
#     all_subs_report[subj_dir.name] = report
# save_results(all_subs_report)
all_subs_report = load_data("results/20260128_110832_allsubs_stationarity/data/saved_data.pkl")

#%% process ECN
# results = {}
# for lag in range(maxlag+1): # 0,...,maxlag
#     results[lag] = {
#         "AD":  {"strengths": [], "probs": []},
#         "FTD": {"strengths": [], "probs": []},
#         "CN":  {"strengths": [], "probs": []}}

# for subj_dir in subjects:
#     subj_id = subj_dir.name # i.e. 'sub-xxx'
#     subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. int('xxx')
#     print(f"\n- {subj_id} -", end=' ', flush=True)

#     #%% preprocessing: load and segment
#     filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
#     eeg, _, channels = load_eeg(filepath, resample=fs_res, preload=True) # notice resampling!
#     epochs = split_epochs(eeg, n_epochs=n_epochs) # split into 10 equal segments
        
#     #%% lingam
#     sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks
#     # lingam+bootstrap
#     ling_strength, ling_probs = lingam_ecn_boot(epochs, channels, maxlag, current_subject=sub)
#     #***ling_strength = lingam_ecn(epochs, channels, maxlag, current_subject=sub)

#     for lag in range(maxlag+1): # 0,...,maxlag
#         results[lag][subj_group]["strengths"].append(ling_strength[lag])
#         results[lag][subj_group]["probs"].append(ling_probs[lag])

# save_results(results)
results = load_data("results/20260215_154600_ling-bootstrap_minefx0_2_plots_th80_prob95/data/saved_data.pkl")

#%% plot ECNs for each group
ch_names = results[0]["AD"]["strengths"][0].columns # remind indexes' names = columns' names
n_ch = len(ch_names)
pos = {"CN":0,"FTD":1,"AD":2}
thresh_pct=80 # strengths threshold (percentile)
min_prob = 0.95 # ~alpha=0.05
n_lags = len(results) # istantaneous+lags count

# aggragate strengths and probs: avg across lags and subjects for each group
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

# for pretty plots
all_strengths, all_n_probs = [], []
max_n_pr, min_n_pr = 0, math.inf # min and max for min-max norm [0,1]
for group, ecn in res_groups.items(): 
    s = np.abs(ecn["strength"])
    all_strengths.append(s.ravel()) # 2D->1D

    p = ecn["n_probs"]
    if np.min(p) < min_n_pr: min_n_pr = np.min(p)
    if np.max(p) > max_n_pr: max_n_pr = np.max(p)
all_strengths = np.concatenate(all_strengths)
min_thresh = np.percentile(all_strengths, thresh_pct) # global threshold for strengths!


# plot results for each group
fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True) 
for group, ecn in res_groups.items():
        s = np.abs(ecn["strength"])
        
        p = ecn["n_probs"]
        p_norm = (p-min_n_pr) / (max_n_pr-min_n_pr) # min-max norm [0,1]

        strength_group_df = pd.DataFrame(s, index=ch_names, columns=ch_names)
        p_norm_group_df = pd.DataFrame(p_norm, index=ch_names, columns=ch_names)
        
        plot_ecn(strength_group_df, min_thresh, ax=axes[pos[group]], title=group, widths=p_norm_group_df)

fig.suptitle(f"LiNGAM (prob_min={min_prob}, th_pct={thresh_pct})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results(results) # save figures