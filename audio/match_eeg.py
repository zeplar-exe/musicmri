from collections import defaultdict
import mne
import glob
import os
from itertools import product
import json
import librosa
import audio.cluster as cluster

def analyze_3774():
    MODEL = "ds003774-raw-chunk-band"
    SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008", "sub-009", "sub-010"]
    SESSIONS = ["ses-01", "ses-02", "ses-03", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", "ses-12"]
    LISTEN_DURATIONS = {
        "ses-01": 125, "ses-02": 114, "ses-03": 132, "ses-04": 111, "ses-05": 124, "ses-07": 116,
        "ses-08": 121, "ses-09": 126, "ses-10": 197, "ses-11": 113, "ses-12": 146,
    }
    
    stimuli_data = {}
    for file in glob.glob("data/data-raw/ds003774/stimuli/*.mp3"):
        stimuli_data[os.path.basename(file)] = defaultdict(list)

    with open(f"models/{MODEL}/clusters.json", "w") as f:
        clusters = json.load(f)
        
        for cluster_id, cluster_data in clusters.items():
            for instance in cluster_data:
                stimulus_id = instance["file"]
                if stimulus_id not in stimuli_data:
                    print(f"Warning: file {stimulus_id} not found in stimuli_data for {cluster_id}.")
                    continue
                stimuli_data[stimulus_id][cluster_id].append(instance)
    
        for subject, session in product(SUBJECTS, SESSIONS):
            folder = f"data/data-raw/ds003774/{subject}/{session}/{subject}/eeg/"
            set_file = glob.glob(os.path.join(folder, "*.set"))[0]
            raw = mne.io.read_raw_eeglab(set_file, preload=True)
            
            print(raw.ch_names)
            
            relevant_instances = defaultdict(list)
            
            for cluster_id, instances in stimuli_data[session].items():
                for instance in instances:
                    if instance["start_time"] < LISTEN_DURATIONS[session]:
                        relevant_instances[cluster_id].append(instance)
            
            psds = defaultdict(list)
            averages = defaultdict(list)
            crops = defaultdict(list)
            
            global_data = raw.get_data()
            global_psd = mne.time_frequency.psd_array_multitaper(global_data, sfreq=raw.info["sfreq"], fmin=0.5, fmax=100, n_jobs=1)
            global_average = global_data.mean(axis=1)
            
            for cluster_id, instances in relevant_instances.items():
                for instance in instances:
                    cropped_raw = raw.copy().crop(tmin=instance["start_time"], tmax=instance["end_time"])
                    cropped_data = cropped_raw.get_data()
                    psd = mne.time_frequency.psd_array_multitaper(cropped_raw, sfreq=raw.info["sfreq"], fmin=0.5, fmax=100, n_jobs=1)
                    psds[cluster_id].append(psd)
                    averages[cluster_id].append(cropped_data.mean(axis=1))
                    crops[cluster_id].append(cropped_data)
            
                # determine whether there is a unique EEG signature across the cluster... how do that?
                    # naive: pearson coefficient between the signals intracluster + coefficients with extracluster signals
                    # perhaps, train a cluster classifier and log accuracy; would have to train across all subjects and stimuli, and then average the results
                    # also, Claude is telling me about Spearman's which can catch quadratics???
                    # or: compare the power spectrum of the cluster signals to the power spectrum of the extracluster signals; if there is a unique signature, then there should be a significant difference in the power spectrum
                    # and just: compare average to extracluster average
                # might want to preload all of the stimuli
            
            # should figure out which eeg nodes are auditory/relevant; time to read
            # condition 1: samples must be generally similar
            # condition 2: samples must be different from the rest of the set
            # get the variance of the averages... if small enough, C1 met
            # ...uh, what statistical test is even relevant here?


if __name__ == "__main__":
    analyze_3774()