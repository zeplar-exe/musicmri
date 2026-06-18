"""Shared ds003774 EEG wiring: dataset layout, preprocessing, epoch cropping.

Both match_eeg.py (correlation/null analysis) and visualize_evoked.py (evoked
butterfly plots) import from here so the epochs they work on are extracted
identically -- same channels, same filter, same sample grid.
"""
from collections import defaultdict
import glob
import os
import mne
import numpy as np

mne.set_log_level("ERROR")

SUBJECTS = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008", "sub-009", "sub-010",
    "sub-011", "sub-012", "sub-013", "sub-014", "sub-015", "sub-016", "sub-017", "sub-018", "sub-019", "sub-020",
]
SESSIONS = ["ses-01", "ses-02", "ses-03", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", "ses-12"]
LISTEN_DURATIONS = {
    "ses-01": 125, "ses-02": 114, "ses-03": 132, "ses-04": 111, "ses-05": 124, "ses-06": 100,
    "ses-07": 116, "ses-08": 121, "ses-09": 126, "ses-10": 197, "ses-11": 113, "ses-12": 117,
}
ERP_LAG = 0.25      # seconds; how much lag before an ERP response?
ERP_WINDOW = 1.25   # seconds; how long should the ERP response last?
BANDPASS = (1.0, 40.0)  # high-pass kills slow drift, low-pass kills line and artifacts
RESAMPLE_HZ = 100   # post-filter; >2x the 40 Hz band, ~10x fewer samples downstream
# Peak-to-peak artifact rejection: drop any epoch whose worst channel swings more
# than this (volts) -- blinks/movement/boundary discontinuities reach mV, ~100x
# physiological. Shared so the correlation/null stats and the evoked plots are
# built from the same cleaned epochs. Set to None to disable.
REJECT_PTP = 100e-6  # volts (100 µV); still well above physiological alpha (~20-50 µV ptp)
# Vertex / fronto-central cluster. The auditory N1/P2 is a TANGENTIAL dipole on
# the supratemporal plane: its axis points up, so the scalp field peaks at the
# vertex (Cz) and inverts at the mastoids, while the lateral temporal scalp sits
# on the dipole's near-zero equator. So we sample the TOP of the head, not the
# ears -- the inner cap around Cz (E129), z >= 8.5, symmetric L/R. Broaden toward
# Fz (E5, E11, E12) if N1 turns out to sit more anterior.
VERTEX_CHANNELS = [
    "E129", "E55", "E7", "E106", "E31", "E80", "E54", "E79",
    "E6", "E37", "E87", "E30", "E105", "E13", "E112",
]


def load_subject_epochs(subject, session):
    """Load one subject/session, return (present_channels, crop_at) or None.

    `present_channels` is the temporal subset actually in this recording.
    `crop_at(onset_seconds)` returns a (n_channels, n_window) array for the
    window starting at `onset_seconds`, or None if it runs off the recording.
    The caller adds ERP_LAG itself (the null path crops at raw onsets).
    """
    folder = f"data/data-raw/ds003774/{subject}/{session}/eeg/"
    set_file = glob.glob(os.path.join(folder, "*.set"))
    if not set_file:
        return None

    raw = mne.io.read_raw_eeglab(set_file[0], preload=True)

    # The 40 Hz low-pass already kills 50/60 Hz line noise, so a separate notch
    # was redundant -- dropped. (If BANDPASS's high end is ever raised above the
    # mains freq, add a notch back.)
    # NOTE: no n_jobs here. match_eeg imports the TF/Keras/UMAP stack via
    # `cluster`; on macOS (spawn) n_jobs>1 makes every worker re-import all of
    # it, which is far slower than the filter itself.
    raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], verbose="ERROR")

    # Re-reference BEFORE picking. The recording is Cz-referenced (E129 reads
    # flat), which subtracts the vertex -- exactly where the auditory response is
    # maximal -- out of every channel; near-vertex electrodes then cancel against
    # the reference. Averaging over the FULL 129-channel montage recovers the true
    # average reference (the Cz term cancels out), so the vertex deflection
    # survives and E129 becomes a real channel again. Requires every channel still
    # present, hence reref then pick.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    present = [ch for ch in VERTEX_CHANNELS if ch in raw.ch_names]
    if not present:
        return None
    raw.pick(present)

    # Band-limited to 40 Hz, so 100 Hz sampling is ample (Nyquist 50). Shrinks
    # every epoch ~Nx, cutting crop/correlation/null cost downstream.
    raw.resample(RESAMPLE_HZ, verbose="ERROR")

    full = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_win = int(round(ERP_WINDOW * sfreq))
    n_samp = full.shape[1]

    def crop_at(onset):
        s0 = int(round(onset * sfreq))
        if s0 < 0 or s0 + n_win > n_samp:
            return None
        window = full[:, s0:s0 + n_win]
        if REJECT_PTP is not None and np.ptp(window, axis=1).max() > REJECT_PTP:
            return None  # gross artifact -- drop the whole epoch
        return window

    return present, crop_at
