from scipy.signal import find_peaks, butter, filtfilt
import numpy as np

def standardize_features(method, features):
    features2 = features.copy()
    if method == "min-max":
        subtract = np.nanmin(features, axis=1)
        divide = np.nanmax(features, axis=1) - subtract
    if method == "z-norm":
        subtract = np.nanmean(features, axis=1)
        divide = np.nanstd(features, axis=1) - subtract
    for i in range(len(features)):
        features2[i] = (features[i] - subtract[i]) / divide[i]

    return features2

def highpass(x, cutoff=3/4, fs=28, order=3):
    """Return a low-pass filtered signal."""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalize the cutoff
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # Filter coefficients
    y = filtfilt(b, a, x)  # Filter the signal
    return np.array(y).astype(np.float32)

def lowpass(x, cutoff=4, fs=28, order=3):
    """Return a low-pass filtered signal."""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalize the cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Filter coefficients
    y = filtfilt(b, a, x)  # Filter the signal
    return np.array(y).astype(np.float32)

def bandpass(x, low_freq_cut, high_freq_cutoff, fs=28):
    x = lowpass(x, cutoff=high_freq_cutoff, fs=fs)
    x = highpass(x, cutoff=low_freq_cut, fs=fs)

    return x

def detect_peaks(x, fs, low_freq_cut, high_freq_cutoff):
    x = lowpass(x, cutoff=high_freq_cutoff, fs=fs)
    x = highpass(x, cutoff=low_freq_cut, fs=fs)

    min_distance = int(0.66*fs)
    peaks, _ = find_peaks(x, distance=min_distance)

    return peaks

def find_closest_neighbor(x1, x2, max_distance):
    candidates = []
    for i in range(len(x1)):
        x2_candidates = []
        for j in range(len(x2)):
            if np.abs(x1[i]-x2[j]) < max_distance:
                x2_candidates.append(x2[j])
        candidates.append(x2_candidates)

def compare_peaks(x1, x2):
    x1_diff = np.diff(x1)
    x1_median_diff = np.median(x1_diff)
    x2_diff = np.diff(x2)
    x2_median_diff = np.median(x2_diff)

