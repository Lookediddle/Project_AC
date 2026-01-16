#%% imports
from functions import *
from utils.io import *

#%% load and segment
filepath = r"../dataset/derivatives/sub-017/eeg/sub-017_task-eyesclosed_eeg.set"
eeg, fs, channels = load_eeg(filepath, preload=True)

epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
print(epochs.shape)

#%% granger causality
mean_pvals, binary_adj = granger_ecn(epochs, maxlag=6, alpha=0.05)

#%% load results
#channels, mean_pvals, binary_adj = load_data("results/20260116_095015/data/saved_data.pkl")

#%% plot ECNs
mean_pvals.rename(
    index=lambda x: channels[int(x.split("_")[0])],
    columns=lambda x: channels[int(x.split("_")[0])],
    inplace=True
)
binary_adj.rename(
    index=lambda x: channels[int(x.split("_")[0])],
    columns=lambda x: channels[int(x.split("_")[0])],
    inplace=True
)
plot_ecn(mean_pvals, binary_adj, channel_order=CHANNEL_ORDER, threshold=2.0,  # p < 0.01
    title="Granger ECN"
)


#%% save results
res={"channels":channels, "mean_pvals":mean_pvals, "binary_adj":binary_adj}
save_results(res)