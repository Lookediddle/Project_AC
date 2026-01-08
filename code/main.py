#%% imports
import mne

#%%
eeg_file = r"../dataset/derivatives/sub-002/eeg/sub-002_task-eyesclosed_eeg.set"
eeg = mne.io.read_raw_eeglab(eeg_file, preload=True)
print(eeg.info)
eeg.plot()

eeg_file = r"../dataset/derivatives/sub-004/eeg/sub-004_task-eyesclosed_eeg.set"
eeg = mne.io.read_raw_eeglab(eeg_file, preload=True)
print(eeg.info)
eeg.plot()

#%%
from utils.io import save_all_open_figures
save_all_open_figures()