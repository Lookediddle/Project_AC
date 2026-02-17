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
print('--- check stationarity ---')
all_subs_report = {}
for subj_dir in subjects:
    print(f"\n- {subj_dir.name} -", end=' ', flush=True)
    report = analyze_subject_stationarity(subj_dir, resample=fs_res, n_epochs=n_epochs)
    all_subs_report[subj_dir.name] = report
save_results(all_subs_report)
#all_subs_report = load_data("results/20260128_110832_allsubs_stationarity/data/saved_data.pkl") # use saved data for speed!

#%% process ECN
print('--- Granger causality ---')
results = {}
for lag in range(1,maxlag+1):
    results[lag] = {
        "AD":  {"pvals": []},
        "FTD": {"pvals": []},
        "CN":  {"pvals": []}}

for subj_dir in subjects:
    subj_id = subj_dir.name # i.e. 'sub-xxx'
    subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. subs_to_groups[int('xxx')]
    print(f"\n- {subj_id} -", end=' ', flush=True)

    #%% preprocessing: load and segment
    filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
    eeg, _, channels = load_eeg(filepath, resample=fs_res, preload=True) # notice resampling!
    epochs = split_epochs(eeg, n_epochs=n_epochs) # split into 10 equal segments
    
    #%% granger 
    curr_sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks
    gran_pvals = granger_ecn(epochs, channels, maxlag, curr_sub)
    
    for lag in range(1,maxlag+1):
        results[lag][subj_group]["pvals"].append(gran_pvals[lag])

save_results(results)
#results = load_data("results/20260215_154921_gran_plots_th80_alpha0_05_agg1/data/saved_data.pkl") # use saved data for speed!

#%% aggregate ECNs for each group
max_pval = alpha # if close to 0, only most certain causal links are considered
strengths_groups = aggregate_granger(results, max_pval) # empirical strengths in [0,1] (proportional to lag contribute)

#%% plot ECNs for each group
ch_names = results[1]["AD"]["pvals"][0].columns # remind indexes' names = columns' names
pos = {"CN":0,"FTD":1,"AD":2}
thresh_pct=80 # strengths threshold (percentile) 

all_strengths, all_n_probs = [], []
for group, ecn in strengths_groups.items(): 
    s = np.abs(ecn["strength"])
    all_strengths.append(s.values.ravel()) # 2D->1D
all_strengths = np.concatenate(all_strengths)
min_thresh = np.percentile(all_strengths, thresh_pct) # global threshold for strengths!

# plot results for each group
fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True) 
for group, ecn in strengths_groups.items():
        s = np.abs(ecn["strength"]) # already in [0,1]
        strength_group_df = pd.DataFrame(s, index=ch_names, columns=ch_names)
        plot_ecn(strength_group_df, min_thresh, ax=axes[pos[group]], title=group, widths=strength_group_df)

fig.suptitle(f"Granger (pval_max={max_pval}, th_pct={thresh_pct})", fontsize=16)
plt.show()

save_results(results) # save figures