from collections import defaultdict
import mne
import os
from itertools import combinations
import json
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import cluster as cluster
from eeg_3774 import (
    SUBJECTS, SESSIONS, LISTEN_DURATIONS, ERP_LAG, ERP_WINDOW,
    load_subject_epochs,
)
import math
import sys
import warnings

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", message=".*boundary.*")
warnings.filterwarnings("ignore", message=".*annotation.*")


def _pair_stats(epochs):
    chan_epochs = defaultdict(list)
    
    for epoch in epochs:
        for ch, vec in epoch.items():
            chan_epochs[ch].append(vec)

    channel_means, channel_msq = [], []
    for vecs in chan_epochs.values():
        if len(vecs) < 2:
            continue
        
        t = min(len(v) for v in vecs)
        m = np.stack([v[:t] for v in vecs])
        m = m - m.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(m, axis=1)
        
        m = m[norms > 0] / norms[norms > 0, None]
        n = m.shape[0]
        
        if n < 2:
            continue
        
        s = m.sum(axis=0)
        channel_means.append((s @ s - n) / (n * (n - 1)))
        # ||U U^T||_F^2 == ||U^T U||_F^2, so use the smaller Gram. When epochs
        # outnumber timepoints (dense clusters firing dozens of times), the T x T
        # form keeps this ~linear in N instead of quadratic -- the difference
        # between a 3-hour run and a short one on dense models.
        # > Thanks Claude
        gram = m @ m.T if n <= m.shape[1] else m.T @ m
        channel_msq.append(((gram * gram).sum() - n) / (n * (n - 1)))

    if not channel_means:
        return None, None
    return float(np.mean(channel_means)), float(np.sqrt(max(np.mean(channel_msq), 0.0)))


def _prep_null_pool(null_bank):
    per_channel = defaultdict(dict)
    pool_sizes = {}
    
    for subject, epochs in null_bank.items():
        pool_sizes[subject] = len(epochs)
        chan_rows = defaultdict(list)
        for ep in epochs:
            for ch, vec in ep.items():
                chan_rows[ch].append(vec)
        for ch, rows in chan_rows.items():
            m = np.stack(rows).astype(float)
            m = m - m.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            per_channel[ch][subject] = m / norms
    
    return per_channel, pool_sizes


def _null_rms(per_channel, pool_sizes, occ, n_perm, rng):
    subjects = [s for s, n in pool_sizes.items() if n >= occ]
    
    if len(subjects) < 2:
        return np.array([])

    out = np.empty(n_perm)
    for p in range(n_perm):
        draws = {s: rng.choice(pool_sizes[s], occ, replace=False) for s in subjects}

        channel_msq = []
        for subj_mats in per_channel.values():
            rows = [subj_mats[s][draws[s]] for s in subj_mats if s in draws]
            if not rows:
                continue
            u = np.concatenate(rows, axis=0)
            n = u.shape[0]
            if n < 2:
                continue
            gram = u @ u.T if n <= u.shape[1] else u.T @ u
            channel_msq.append(((gram * gram).sum() - n) / (n * (n - 1)))

        out[p] = np.sqrt(max(np.mean(channel_msq), 0.0)) if channel_msq else np.nan

    return out[~np.isnan(out)]

# future setup should coalesce all usages of a single model so we can get a single null result and r value per cluster
def analyze_3774(model_dir):
    NULL_POOL = 100 # random epochs cropped per subject to draw the null from
    NULL_PERM = 1000 # permutations per distinct epoch count

    rng = np.random.default_rng(0)
    model_name = os.path.basename(model_dir)

    os.makedirs(f"correlates/ds003774/{model_name}", exist_ok=True)
    predictions_path = f"correlates/ds003774/{model_name}/predictions.json"

    if os.path.exists(predictions_path):
        with open(predictions_path) as g:
            predictions = defaultdict(list, json.load(g))
        trimmed = sum(
            1
            for rows in predictions.values()
            for i, m in enumerate(rows)
            if abs(m["start_time"] - i * cluster.CHUNK_DURATION) > 1e-6
        )
        print(f"Reusing cached predictions ({trimmed} chunks carry a silence-trim offset). Delete {predictions_path} to force re-clustering.")
        if trimmed == 0:
            print("  WARNING: no trim offsets found -- this file may predate the silence-trim. Delete it to regenerate if unsure.")
    else:
        predictions = defaultdict(list)
        encoder, pca, clusterer, norm = cluster._load_models(model_dir)

        stimulus_foler = "data/data-raw/ds003774/stimuli"
        for mus_file in tqdm(os.listdir(stimulus_foler), desc="Predicting clusters for stimuli"):
            samples, onset_offsets = cluster._chunks_with_onsets(os.path.join(stimulus_foler, mus_file))

            if len(samples) == 0:
                continue

            labels, _ = cluster._predict_chunks(samples, encoder, pca, clusterer, norm)
            predictions[mus_file] = [{
                "start_time": i * cluster.CHUNK_DURATION + float(onset_offsets[i]),
                "cluster_id": int(label),
            } for i, label in enumerate(labels)]

        with open(predictions_path, "w") as g:
            json.dump(predictions, g, indent=4)

    average_correlations = defaultdict(lambda: defaultdict(dict))
    pearsons = defaultdict(lambda: defaultdict(list))
    aggregated = defaultdict(dict)
    null_pvalues = defaultdict(dict)

    with tqdm(total=len(SESSIONS)*len(SUBJECTS) + 1, desc="Analyzing EEG data") as pbar:
        for session in SESSIONS:
            matches = [m for m in predictions[f"{session}.mp3"]
                    if m["cluster_id"] != -1 and m["start_time"] < LISTEN_DURATIONS[session]]
            if not matches:
                continue

            pooled = defaultdict(list)
            null_bank = {}

            for subject in SUBJECTS:
                loaded = load_subject_epochs(subject, session)
                if loaded is None:
                    continue
                present, crop_at = loaded

                segments = defaultdict(list)
                for m in matches:
                    data = crop_at(m["start_time"] + ERP_LAG)
                    
                    if data is None:
                        continue
                    
                    cid = int(m["cluster_id"])
                    segments[cid].append(data)
                    pooled[cid].append(dict(zip(present, data)))

                for cluster_id, segs in segments.items():
                    if len(segs) < 2:
                        continue
                    
                    results = []
                    for a, b in combinations(segs, 2):
                        min_t = min(a.shape[1], b.shape[1])
                        a, b = a[:, :min_t], b[:, :min_t]
                        ch_rs = [pearsonr(a[c], b[c]).statistic for c in range(a.shape[0])]
                        ch_rs = [r for r in ch_rs if not math.isnan(r)]
                        if not ch_rs:
                            continue
                        results.append({"r": sum(ch_rs) / len(ch_rs)})
                        
                    if not results:
                        continue
                    
                    correlations = [p["r"] for p in results]
                    pearsons[f"{subject}-{session}"][cluster_id].append(results)
                    average_correlations[f"{subject}-{session}"][cluster_id] = {
                        "average": sum(correlations) / len(correlations),
                        "average_abs": sum(abs(c) for c in correlations) / len(correlations),
                        "num_pairs": len(correlations),
                    }

                max_onset = LISTEN_DURATIONS[session] - ERP_WINDOW
                bank = []
                for onset in rng.uniform(0.0, max_onset, size=NULL_POOL):
                    data = crop_at(onset)
                    if data is not None:
                        bank.append(dict(zip(present, data)))
                if bank:
                    null_bank[subject] = bank
                
                pbar.update(1)

            n_subjects = sum(1 for s in SUBJECTS if s in null_bank)
            if n_subjects < 2:
                continue

            null_prep, pool_sizes = _prep_null_pool(null_bank)
            null_cache = {}
            
            for cluster_id, epochs in clusters_sorted(pooled):
                if len(epochs) < 2:
                    continue
                signed, rms = _pair_stats(epochs)
                if rms is None:
                    continue

                occ = max(1, round(len(epochs) / n_subjects))
                
                if occ not in null_cache:
                    null_cache[occ] = _null_rms(null_prep, pool_sizes, occ, NULL_PERM, rng)
                
                null = null_cache[occ]

                entry = {
                    "average": signed,
                    "rms_r": rms,
                    "num_epochs": len(epochs),
                }
                aggregated[session][cluster_id] = entry

                if null.size:
                    p = (1 + int(np.sum(null >= rms))) / (1 + null.size)
                    null_pvalues[session][cluster_id] = {
                        "rms_r": rms,
                        "p": p,
                        "null_mean": float(null.mean()),
                        "null_p95": float(np.percentile(null, 95)),
                        "num_epochs": len(epochs),
                        "n_perm": int(null.size),
                    }
                
            pbar.update(1)

    with open(f"correlates/ds003774/{model_name}/average_correlations.json", "w") as h:
        json.dump(average_correlations, h, indent=4)
    with open(f"correlates/ds003774/{model_name}/pearson_correlations.json", "w") as i:
        json.dump(pearsons, i, indent=4)
    with open(f"correlates/ds003774/{model_name}/aggregated_correlations.json", "w") as j:
        json.dump(aggregated, j, indent=4)
    with open(f"correlates/ds003774/{model_name}/null_pvalues.json", "w") as k:
        json.dump(null_pvalues, k, indent=4)


def clusters_sorted(pooled):
    """Iterate clusters by epoch count so null_cache fills predictably."""
    return sorted(pooled.items(), key=lambda kv: len(kv[1]))


if __name__ == "__main__":
    analyze_3774(sys.argv[1])
