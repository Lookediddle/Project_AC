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
#results = load_data("results/20260130_141033_4_lags_var-ling_allsubs/data/saved_data.pkl")

#%% plot ECNs for each group
ch_names = results["AD"]["strengths"][0].columns # remind indexes' names = columns' names
pos = {"CN":0,"FTD":1,"AD":2}
thresh=10000 # scegliere soglia sensata

fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
for group, all_ecns in results.items():
    strength_group = np.zeros_like(results[group]["strengths"][0])

    # keep highest causal links
    for subj in range(0,len(all_ecns["strengths"])):
        strength_group = np.maximum(
            strength_group, np.abs(all_ecns["strengths"][subj]))
        
    strength_group_df = pd.DataFrame(
            strength_group, index=ch_names, columns=ch_names)
    
    plot_ecn(strength_group_df, thresh, ax=axes[pos[group]], title=group)

fig.suptitle(f"LiNGAM (highest vals, th={thresh})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results() # save figures