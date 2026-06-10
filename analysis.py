from collections import defaultdict

import numpy as np
import mne
import librosa
from datetime import datetime, timedelta
from scipy.stats import pearsonr


def analyze_waveform_eeg(eeg_file: str, audio_file: str):
    WINDOW_S = 1
    
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    channels: list[str] = raw.ch_names
    duration: float = raw.duration
    start_datetime: datetime = raw.info["meas_date"]
    
    channel_sums = defaultdict(lambda: defaultdict(list))
    
    for t in np.arange(0, duration - WINDOW_S, WINDOW_S):
        data, times = raw.copy().crop(tmin=t, tmax=t + WINDOW_S).get_data(return_times=True)
        
        for i, channel in enumerate(channels):   
            channel_data = data[i]
            
            # pearsonr on wav and EEG
            # descriptors on-the-fly (with cache) to give labels for correlation
            # it would help to have lags for this... as in, lagged *away* from the onset of the stimulus

def analyze_waveform_mri(mri_file: str, audio_file: str):
    pass

def analyze_feature_eeg(feature_predictions: list[dict], eeg_file: str):
    LAG_WINDOW_S = 0.3
    LAGS = 10
    
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    channels: list[str] = raw.ch_names
    start_datetime: datetime = raw.info["meas_date"]
    
    channel_sums = defaultdict(lambda: defaultdict(list))
    
    for pred in feature_predictions:
        label: float = pred["label"]
        start_time: float = pred["start"]
        end_time: float = pred["end"]
        confidence: float = pred["confidence"]
        
        real_start_time: datetime = start_datetime + timedelta(seconds=start_time)
        real_end_time: datetime = start_datetime + timedelta(seconds=end_time)
        
        real_start_time += timedelta(seconds=0) # need to align with the start of the EEG recording
        real_end_time += timedelta(seconds=0)
        
        for lag in range(LAGS + 1):
            lag_s = lag * LAG_WINDOW_S
            data, times = raw.copy().crop(tmin=real_start_time.second + lag_s, tmax=real_end_time.second + lag_s).get_data(return_times=True)
            
            for i, channel in enumerate(channels):
                channel_data = data[i]
                channel_sum = np.sum(channel_data)
                channel_sums[label][channel].append(channel_sum)

def analyze_feature_mri(feature_predictions: list[dict], mri_file: str):
    pass


def envelope_derivative(y, sr=22050, hop_length=512, frame_length=2048, log=True):
    """Amplitude envelope and its time-derivative for a (cropped) waveform.

    Parameters
    ----------
    y : np.ndarray
        Mono waveform, already loaded and cropped by the caller
        (e.g. librosa.load(...) then sliced to the region of interest).
    sr : int
        Sample rate of y (sets the frame -> seconds mapping).
    hop_length, frame_length : int
        RMS framing. Defaults match the CNN's mel settings, so the envelope
        lands on the same ~43 Hz frame grid (sr / hop_length).
    log : bool
        If True, the envelope is in dB and the derivative is dB/sec -- a
        loudness-relative rate of change, so the same crescendo reads the same
        whether the track is loud or quiet. That stability is what you want for
        binning by derivative range. Set False for linear RMS-per-second.

    Returns
    -------
    times : np.ndarray, shape (n_frames,)
        Frame-center times in seconds, relative to the start of the crop.
    env : np.ndarray, shape (n_frames,)
        Amplitude envelope (dB if log else linear RMS).
    denv : np.ndarray, shape (n_frames,)
        d(env)/dt in units per second. Positive = getting louder (crescendo),
        negative = getting quieter (decrescendo).
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length)[0]

    env = librosa.amplitude_to_db(rms, ref=np.max) if log else rms
    times = librosa.frames_to_time(
        np.arange(len(env)), sr=sr, hop_length=hop_length)

    # Per-second derivative. np.gradient uses centered differences in the
    # interior and one-sided differences at the edges, dividing by the actual
    # time spacing, so denv is already in units/sec. Needs >= 2 frames.
    if len(env) < 2:
        denv = np.zeros_like(env)
    else:
        denv = np.gradient(env, times)

    return times, env, denv
