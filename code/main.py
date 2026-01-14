#%% imports
from functions import *

#%% load and segment
filepath = r"../dataset/derivatives/sub-017/eeg/sub-017_task-eyesclosed_eeg.set"
eeg, fs, ch_names = load_eeg(filepath, preload=True)

epochs = split_epochs(eeg, n_epochs=10) # split into 10 equal segments
print(epochs.shape)

#%% granger causality
mean_pvals, binary_adj = granger_ecn(epochs, maxlag=6, alpha=0.05)

#%% save results
from utils.io import *
res={"eeg":eeg}
save_results(res)