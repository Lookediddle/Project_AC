import mne

eeg_file = r"../dataset/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

eeg = mne.io.read_raw_eeglab(eeg_file, preload=True)

print(eeg.info)

eeg.plot()

print('end')