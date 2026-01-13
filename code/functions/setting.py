import mne
import numpy as np


def load_eeg(filepath, preload=True):
    """
    Load a EEG file (.set).

    Parameters
    ----------
    filepath : str
        Path to .set file
    preload : bool
        Whether to preload data into memory

    Returns
    -------
    eeg_data : ndarray, shape (n_channels, n_samples)
        EEG signal
    sfreq : float
        Sampling frequency (Hz)
    ch_names : list of str
        Channel names
    """

    raw = mne.io.read_raw_eeglab(filepath, preload=preload, verbose=False)
    #print(raw.info) # general infos
    raw.plot()

    eeg_data = raw.get_data() # shape (n_channels, n_samples)
    sfreq = raw.info["sfreq"] # Fs
    ch_names = raw.info["ch_names"] # Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz

    return eeg_data, sfreq, ch_names


import numpy as np


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

