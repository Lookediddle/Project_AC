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
results = {}
for lag in range(maxlag+1): # 0,...,maxlag
    results[lag] = {
        "AD":  {"strengths": [], "probs": []},
        "FTD": {"strengths": [], "probs": []},
        "CN":  {"strengths": [], "probs": []}}

for subj_dir in subjects:
    subj_id = subj_dir.name # i.e. 'sub-xxx'
    subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. int('xxx')
    print(f"\n- {subj_id} -", end=' ', flush=True)

    #%% preprocessing: load and segment
    filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
    eeg, _, channels = load_eeg(filepath, resample=fs_res, preload=True) # notice resampling!
    epochs = split_epochs(eeg, n_epochs=n_epochs) # split into 10 equal segments
        
    #%% lingam
    sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks
    # lingam+bootstrap
    ling_strength, ling_probs = lingam_ecn_boot(epochs, channels, maxlag, current_subject=sub)
    #***ling_strength = lingam_ecn(epochs, channels, maxlag, current_subject=sub)

    for lag in range(maxlag+1): # 0,...,maxlag
        results[lag][subj_group]["strengths"].append(ling_strength[lag])
        results[lag][subj_group]["probs"].append(ling_probs[lag])

save_results(results)
#results = load_data("results/20260204_201717_4_lags_ling-bootstrap_allsubs_no_lags_aggregation/data/saved_data.pkl")

#%% plot ECNs for each group
ch_names = results[0]["AD"]["strengths"][0].columns # remind indexes' names = columns' names
n_ch = len(ch_names)
pos = {"CN":0,"FTD":1,"AD":2}
thresh_pct=75 # strengths threshold (percentile)
min_prob = 0.95 # ~alpha=0.05
n_tau = len(results) # istantaneous+lags count

# aggragate strengths and probs: accumulate accepted (i.e. probable) ones across lags
res_groups = {
    "AD":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
    "FTD": {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))},
    "CN":  {"strength": np.zeros((n_ch, n_ch)), "n_probs": np.zeros((n_ch, n_ch))}}

for lag,all_groups in results.items():
    for group,all_ecns in all_groups.items():
        strength_group_tot = np.zeros((n_ch, n_ch))
        prob_group_tot = np.zeros((n_ch, n_ch))

        # keep highest causal links (i.e. the most certain ones -> prob>=min_prob)
        n_subs = len(all_ecns["strengths"])
        for subj in range(0,n_subs):
            s = all_ecns["strengths"][subj]
            p = all_ecns["probs"][subj]
            prob_bin = (p >= min_prob).astype(int) # binary (1->link; 0->no link)
            strength_filtered = s*prob_bin # keep most probable strengths

            # accumulate for mean
            strength_group_tot += strength_filtered
            prob_group_tot += prob_bin
        # mean across subs
        mean_strength = strength_group_tot / n_subs       
        # accumulate mean lag contribution for group
        lag_contribute = mean_strength / n_tau
        res_groups[group]["strength"] += lag_contribute

        # accumulate number of accepted probs for group  
        res_groups[group]["n_probs"] += prob_group_tot / n_subs

# find global threhsold for plots
all_strengths = []
for group, ecn in res_groups.items(): 
    s = np.abs((ecn["strength"]).to_numpy())
    all_strengths.append(s.ravel()) # from 2D to 1D
all_strengths = np.concatenate(all_strengths)
thresh = np.percentile(all_strengths, thresh_pct) # here it is!

# plot results for each group
fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True) 
for group, ecn in res_groups.items():
        s = np.abs(ecn["strength"])
        p = ecn["n_probs"]
        strength_group_df = pd.DataFrame(s, index=ch_names, columns=ch_names)
        n_probs_group_df = pd.DataFrame(p, index=ch_names, columns=ch_names)

        plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group, widths=strength_group_df)

fig.suptitle(f"LiNGAM (prob_min={min_prob}, th_pct={thresh_pct})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results() # save figures

# fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
# for group, all_ecns in results.items():
#     strength_group = np.zeros_like(results[group]["strengths"][0])

#     # keep highest causal links
#     for subj in range(0,len(all_ecns["strengths"])):
#         strength_group = np.maximum(
#             strength_group, np.abs(all_ecns["strengths"][subj]))
        
#     strength_group_df = pd.DataFrame(
#             strength_group, index=ch_names, columns=ch_names)
    
#     plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group)

# fig.suptitle(f"LiNGAM (highest vals, th={thresh})", fontsize=16)
# #plt.tight_layout() # useless if constrained_layout=True
# plt.show()

# save_results() # save figures