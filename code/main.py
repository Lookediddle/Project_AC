#%% imports
from functions import *
from utils.io import *

#%% load and segment
filepath = r"../dataset/derivatives/sub-017/eeg/sub-017_task-eyesclosed_eeg.set"
eeg, fs, channels = load_eeg(filepath, preload=True)

epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
print(epochs.shape)

#%% granger causality
#gran_pvals, gran_bin_adj = granger_ecn(epochs, channels, maxlag=6, alpha=0.05)

ling_strength, ling_bin_adj = lingam_ecn(epochs, channels, maxlag=6)

#%% load results
channels, gran_pvals, gran_strength, gran_bin_adj = load_data("results/20260120_182302/data/saved_data.pkl")
gran_strength = causal_strength(gran_pvals) # *********CHIEDERE QUALE HANNO USATO********

#%% plot ECNs
plot_ecn(gran_strength, gran_bin_adj, title="Granger ECN", threshold=2.0)  # p < 0.01
plot_ecn(ling_strength, ling_bin_adj, title="LiNGAM ECN", threshold=0.1) # *** sistemare threshold sensata ***
#%% save results
res_granger = {"gran_pvals":gran_pvals, "gran_strength":gran_strength, "gran_bin_adj":gran_bin_adj}
res_lingam = {"ling_strength":ling_strength, "ling_bin_adj":ling_bin_adj}
res={"channels":channels, "granger":res_granger, "lingam":res_lingam}
save_results(res)