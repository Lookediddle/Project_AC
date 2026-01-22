#%% imports
from functions import *
from utils.io import *
from pathlib import Path

#%% check stationarity
print('--- check stationarity ---')
dataset_dir = Path("../dataset/derivatives/")
subjects = sorted([p for p in dataset_dir.iterdir() if p.name.startswith("sub-")])

# all_subs_report = {}
# for subj_dir in subjects:
#     print(f"\n- {subj_dir.name} -", end=' ', flush=True)
#     report = analyze_subject_stationarity(subj_dir, n_epochs=10)
#     all_subjects_report[subj_dir.name] = report
# save_results(all_subs_report)
all_subs_report = load_data("results/20260122_143416_allsubs_stationarity/data/saved_data.pkl")

#%% load and segment
filepath = r"../dataset/derivatives/sub-017/eeg/sub-017_task-eyesclosed_eeg.set"
eeg, _, channels = load_eeg(filepath, preload=True)

epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
print(epochs.shape)

#%% granger 
gran_pvals, gran_bin_adj = granger_ecn(epochs, channels, maxlag=6, alpha=0.05)
gran_strength = causal_strength(gran_pvals) # *********CHIEDERE QUALE HANNO USATO********

#%% lingam
#ling_strength, ling_bin_adj = lingam_ecn(epochs, channels, maxlag=1)
ling_strength, ling_bin_adj = lingam_ecn_no_lags(epochs, channels)

#%% plot ECNs
plot_ecn(gran_strength, title="Granger ECN", threshold=2.0)  # p < 0.01
plot_ecn(ling_strength, title="LiNGAM ECN", threshold=2.0)

#%% save results
res_granger = {"gran_pvals":gran_pvals, "gran_strength":gran_strength, "gran_bin_adj":gran_bin_adj}
res_lingam = {"ling_strength":ling_strength, "ling_bin_adj":ling_bin_adj}
res={"channels":channels, "granger":res_granger, "lingam":res_lingam}
save_results(res)