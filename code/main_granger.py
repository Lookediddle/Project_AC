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

#%% granger preprocessing: check stationarity
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
# for lag in range(1,maxlag+1):
#     results[lag] = {
#         "AD":  {"pvals": []},
#         "FTD": {"pvals": []},
#         "CN":  {"pvals": []}}

# for subj_dir in subjects:
#     subj_id = subj_dir.name # i.e. 'sub-xxx'
#     subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. subs_to_groups[int('xxx')]
#     print(f"\n- {subj_id} -", end=' ', flush=True)

#     #%% preprocessing: load and segment
#     filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
#     eeg, _, channels = load_eeg(filepath, resample=fs_res, preload=True) # notice resampling!
#     epochs = split_epochs(eeg, n_epochs=n_epochs) # split into 10 equal segments
    
#     #%% granger 
#     curr_sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks
#     gran_pvals = granger_ecn(epochs, channels, maxlag, alpha, curr_sub)
    
#     for lag in range(1,maxlag+1):
#         results[lag][subj_group]["pvals"].append(gran_pvals[lag])

# save_results(results)
results = load_data("results/20260203_170946_4_lags_gran_allsubs_no_lags_aggregation/data/saved_data.pkl")

#%% plot ECNs for each group
ch_names = results[1]["AD"]["pvals"][0].columns # remind indexes' names = columns' names
pos = {"CN":0,"FTD":1,"AD":2}
thresh=1 # at least 1 pval across lags 
max_pval = 1e-12 # if 0: only certain causal links are considered

# create strengths: accumulate accepted pvals across lags
strengths_groups = {
    "AD":  {"strength": np.zeros((len(ch_names), len(ch_names)))},
    "FTD": {"strength": np.zeros((len(ch_names), len(ch_names)))},
    "CN":  {"strength": np.zeros((len(ch_names), len(ch_names)))}}

for lag,all_groups in results.items():
    for group,all_ecns in all_groups.items():
        # keep highest causal links (i.e. the most certain ones -> p-value<=max_pval for the entire group!)
        pvals_group_mean_curr_lag = np.mean(all_ecns["pvals"], axis=0)
        strength_group_curr_lag = (np.round(pvals_group_mean_curr_lag,4) <= max_pval).astype(int) # binary (1->link; 0->no link)
        # accumulate strengths for current lag 
        strengths_groups[group]["strength"] += strength_group_curr_lag

# find global min and max for min-max norm [0,1] for pretty plots
max_s, min_s = 0, math.inf
for group, ecn in strengths_groups.items(): 
    s = ecn["strength"]
    if np.min(s) < min_s: min_s = np.min(s)
    if np.max(s) > max_s: max_s = np.max(s)

# plot results for each group
fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
for group, ecn in strengths_groups.items(): 
    s = ecn["strength"]
    s_norm = (s-min_s) / (max_s-min_s) # min-max norm [0,1]

    strength_group_df = pd.DataFrame(s, index=ch_names, columns=ch_names)
    norm_df = pd.DataFrame(s_norm, index=ch_names, columns=ch_names)

    plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group, widths=norm_df)
fig.suptitle(f"Granger, avg groups (pv_group={max_pval}, th=pv_ok_count={thresh})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results() # save figures

# fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
# for group, all_ecns in results.items():
    
#     # keep highest causal links (i.e. the certain ones -> p-value=0 for the entire group!)
#     pvals_group_mean = np.mean(results[group]["pvals"], axis=0)
#     strength_group = (np.round(pvals_group_mean,4) == 0).astype(int) 
        
#     strength_group_df = pd.DataFrame(
#             strength_group, index=ch_names, columns=ch_names)
    
#     plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group)
# fig.suptitle(f"Granger (highest vals, th={thresh})", fontsize=16)
# #plt.tight_layout() # useless if constrained_layout=True
# plt.show()

# save_results() # save figures