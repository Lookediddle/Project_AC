#%% imports
from functions import *
from utils.io import *

#%% load and segment
filepath = r"../dataset/derivatives/sub-017/eeg/sub-017_task-eyesclosed_eeg.set"
eeg, fs, channels = load_eeg(filepath, preload=True)

epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
print(epochs.shape)

#%% granger causality
mean_pvals, binary_adj = granger_ecn(epochs, channels, maxlag=6, alpha=0.05)

#mean_strength, binary_adj = lingam_ecn(epochs, channels, maxlag=6)

#%% load results
channels, mean_pvals, binary_adj = load_data("results/20260116_182920/data/saved_data.pkl")

#%% plot ECNs
plot_ecn(mean_pvals, binary_adj, title="Granger ECN", threshold=2.0)  # p < 0.01

#%% save results
res={"channels":channels, "mean_pvals":mean_pvals, "binary_adj":binary_adj}
save_results(res)