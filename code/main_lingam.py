#%% imports
from functions import *
from utils.io import *
from pathlib import Path

#%% dataset
dataset_dir = Path("../dataset/derivatives/")
subjects = sorted([p for p in dataset_dir.iterdir() if p.name.startswith("sub-")])
# groups reference
ranges = [(range(1, 37), "AD"), (range(37, 66), "CN"),(range(66, 89), "FTD")]
subs_to_groups = {num:label for r,label in ranges for num in r} # e.g. {1:"AD", 2:"AD", ...}

#%% process ECN
# results = {
#     "AD":  {"strengths": [], "bin_adj": []},
#     "FTD": {"strengths": [], "bin_adj": []},
#     "CN":  {"strengths": [], "bin_adj": []}}

# for subj_dir in subjects:
#     subj_id = subj_dir.name # i.e. 'sub-xxx'
#     subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. int('xxx')
#     print(f"\n- {subj_id} -", end=' ', flush=True)

#     #%% preprocessing: load and segment
#     filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
#     eeg, _, channels = load_eeg(filepath, preload=True)
#     epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
        
#     #%% lingam
#     #ling_strength, ling_bin_adj = lingam_ecn(epochs, channels, maxlag=1)
#     ling_strength, ling_bin_adj = lingam_ecn_no_lags(epochs, channels)

#     results[subj_group]["strengths"].append(ling_strength)
#     results[subj_group]["bin_adj"].append(ling_bin_adj)

# save_results(results)
results = load_data("results/20260122_223151_no_lags_ling_allsubs/data/saved_data.pkl")

#%% plot ECNs ***da sistemare per i groups***
ch_names = results["AD"]["strengths"][0].columns # remind indexes' names = columns' names
pos = {"CN":0,"FTD":1,"AD":2}
thresh=10

fig, axes = plt.subplots(1, 3, figsize=(13, 6), constrained_layout=True)
for group, all_ecns in results.items():
    strength_group = np.zeros_like(results[group]["strengths"][0])
    
    for subj in range(0,len(all_ecns["strengths"])):
        strength_group = np.maximum(
            strength_group, np.abs(all_ecns["strengths"][subj]))
    
    strength_group_df = pd.DataFrame(
            strength_group, index=ch_names, columns=ch_names)
    
    plot_ecn(strength_group, thresh, ax=axes[pos[group]], title=group)

fig.suptitle(f"LiNGAM (highest vals, th={thresh})", fontsize=16)
#plt.tight_layout() # useless if constrained_layout=True
plt.show()

save_results() # save figures