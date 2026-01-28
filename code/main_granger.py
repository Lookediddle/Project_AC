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
# results = {
#     "AD":  {"pvals": [], "bin_adj": []},
#     "FTD": {"pvals": [], "bin_adj": []},
#     "CN":  {"pvals": [], "bin_adj": []}}

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
#     gran_pvals, gran_bin_adj = granger_ecn(epochs, channels, maxlag, alpha, curr_sub)
    
#     results[subj_group]["pvals"].append(gran_pvals)
#     results[subj_group]["bin_adj"].append(gran_bin_adj)

# save_results(results)
results = load_data("results/20260128_132041_4_lags_gran_allsubs_resample/data/saved_data.pkl")

# # apply Bonferroni correction #****da cancellare secondo me****
# alpha_new = alpha / maxlag # alpha = alpha/m where m=#tests (i.e. lags)
# for group, all_ecns in results.items():
#     for subj in range(0,len(all_ecns["pvals"])):
#         pvals = all_ecns["pvals"][subj]
#         binary_adj_new = (pvals == alpha_new).astype(int)
#         results[group]["bin_adj"][subj] = binary_adj_new

#%% plot ECNs for each group
ch_names = results["AD"]["pvals"][0].columns # remind indexes' names = columns' names
pos = {"CN":0,"FTD":1,"AD":2}
thresh=1 

fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
for group, all_ecns in results.items():
    
    # keep highest causal links (i.e. the certain ones -> p-value=0 for the entire group!)
    pvals_group_mean = np.mean(results[group]["pvals"], axis=0)
    strength_group = (np.round(pvals_group_mean,4) == 0).astype(int) 
        
    strength_group_df = pd.DataFrame(
            strength_group, index=ch_names, columns=ch_names)
    
    plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group)
fig.suptitle(f"Granger (highest vals, th={thresh})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results() # save figures