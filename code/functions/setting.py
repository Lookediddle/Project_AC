import mne
import numpy as np
import pandas as pd
from .granger import make_stationary

def load_eeg(filepath, resample=None, preload=True):
    """
    Load a EEG file (.set).

    Parameters
    ----------
    filepath : str
        Path to .set file
    resample : int
        Target frequency for resampling (Hz).
    preload : bool
        Whether to preload data into memory

    Returns
    -------
    eeg_data : ndarray, shape (n_channels, n_samples)
        EEG signal
    sfreq : float
        Sampling frequency (Hz)
    channels : dict
        Channel numbers mapped to names (e.g. {0:"Fp1", 1:"Fp2", etc.})
    """

    raw = mne.io.read_raw_eeglab(filepath, preload=preload, verbose=False)
    #print(raw.info) # general infos
    #raw.plot()

    if resample is not None:
        raw = raw.resample(sfreq=resample, npad="auto")
    eeg_data = raw.get_data() # shape (n_channels, n_samples)
    #print(raw.info) # general infos
    #raw.plot()
    sfreq = raw.info["sfreq"] # Fs
    ch_names = raw.info["ch_names"] # Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz
    
    channels = {}
    for ch, name in enumerate(ch_names):
        channels[ch] = name # e.g. {0:"Fp1", 1:"Fp2", etc.}
    
    return eeg_data, sfreq, channels

def split_epochs(eeg_data, n_epochs=10):
    """
    Split continuous EEG into equal-length epochs.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
        Continuous EEG
    n_epochs : int
        Number of segments to split into

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples_epoch)
    """

    n_channels, n_samples = eeg_data.shape
    samples_per_epoch = n_samples // n_epochs

    epochs = []

    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch = eeg_data[:, start:end]
        epochs.append(epoch)

    return np.stack(epochs)


# stationarity check
def analyze_subject_stationarity(subject_dir, resample=50, n_epochs=10):
    """
    Analyze stationarity for one subject EEG. 
    The EEG is resampled (default at 50 Hz) and divided into epochs (default 10)
    before analyzing stationarity.

    Returns
    -------
    subject_report : dict
        Detailed stationarity report
    """

    eeg_file = list((subject_dir / "eeg").glob("*_eeg.set"))[0]

    eeg, fs, channels = load_eeg(eeg_file, resample=resample, preload=True) # notice resampling!
    epochs = split_epochs(eeg, n_epochs=n_epochs)

    ch_names = [name for _, name in channels.items()]

    subject_report = {
        "subject": subject_dir.name,
        "fs": fs,
        "n_epochs": epochs.shape[0],
        "channels": ch_names,
        "epochs": []
    }

    for e, epoch in enumerate(epochs):
        print(f"... [epoch {e}]", end='-->', flush=True)

        epoch_df = pd.DataFrame(epoch.T, columns=ch_names)

        # stationarity
        epoch_df_stat, n_diffs, diffed_channels = make_stationary(epoch_df)

        epoch_info = {
            "epoch": e,
            "n_diffs": n_diffs,
            "non_stationary_channels": diffed_channels
        }

        subject_report["epochs"].append(epoch_info)

    return subject_report

