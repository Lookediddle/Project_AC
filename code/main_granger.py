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

#%% granger preprocessing: check stationarity
# print('--- check stationarity ---')
# all_subs_report = {}
# for subj_dir in subjects:
#     print(f"\n- {subj_dir.name} -", end=' ', flush=True)
#     report = analyze_subject_stationarity(subj_dir, n_epochs=10)
#     all_subjects_report[subj_dir.name] = report
# save_results(all_subs_report)
all_subs_report = load_data("results/20260122_143416_allsubs_stationarity/data/saved_data.pkl")

#%% process ECN
results = {
    "AD":  {"pvals": [], "bin_adj": []},
    "FTD": {"pvals": [], "bin_adj": []},
    "CN":  {"pvals": [], "bin_adj": []}}

for subj_dir in subjects:
    subj_id = subj_dir.name # i.e. 'sub-xxx'
    subj_group = subs_to_groups[int(subj_id[-3:])] # i.e. subs_to_groups[int('xxx')]
    print(f"\n- {subj_id} -", end=' ', flush=True)

    #%% preprocessing: load and segment
    filepath = list((subj_dir / "eeg").glob("*_eeg.set"))[0]
    eeg, _, channels = load_eeg(filepath, preload=True)
    epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
    
    #%% granger 
    curr_sub = all_subs_report[subj_id] # to skip unnecessary stationarity checks
    gran_pvals, gran_bin_adj = granger_ecn(epochs, channels, 4, 0.05, curr_sub)
    
    results[subj_group]["pvals"].append(gran_pvals)
    results[subj_group]["bin_adj"].append(gran_bin_adj)

save_results(results)

#%% plot ECNs
gran_strength = causal_strength(gran_pvals) # ***provare media di bin_adj per ogni gruppo (strength empirica)*** ******CHIEDERE QUALE HANNO USATO********
plot_ecn(gran_strength, title="Granger ECN", threshold=2.0)  # p < 0.01
save_results() # ***da sistemare per i groups***