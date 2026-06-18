"""Evoked-response diagnostics for a model's candidate clusters.

For a given model, finds every (session, cluster) whose permutation p-value
cleared the threshold, re-extracts the same epochs match_eeg used (via
eeg_3774), and writes two figures per candidate:

  *_heatmaps.png -- stacked:
      top    channel x time, color = event-averaged uV (diverging). The grey
             butterfly "unstacked" so every channel is legible; a real response
             is a vertical feature at a fixed latency, a dipole splits red/blue.
      bottom epoch x time, color = RMS across channels per trial (sequential).
             The single-trial ERP image with channels collapsed: a phase-locked
             response is a vertical stripe across epochs, floor-leakage is
             speckle. This is the closest visual twin of rms_r and exposes any
             lingering artifact epoch as a bright row.

  *_sem.png -- per-channel mean +/- SEM butterfly. Where a channel's shaded band
             clears zero the averaged deflection is reliable; where it straddles
             zero it's noise. Judge sustained crossings, not one-sample blips.

Usage:  python visualize_evoked.py <model_name> [p_threshold]

> Thanks Claude
"""
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import eeg_3774 as eeg

P_THRESHOLD = 0.1


def _candidates(null_pvalues, p_threshold):
    """session -> {cluster_id: stats} for clusters under the p threshold."""
    out = defaultdict(dict)
    for session, clusters in null_pvalues.items():
        for cid, stats in clusters.items():
            if stats["p"] < p_threshold:
                out[session][int(cid)] = stats
    return out


def _collect_epochs(session, cluster_ids, predictions):
    """Pool epochs across subjects for the given clusters.

    Returns {cluster_id: [ {channel: 1d array}, ... ]} -- one dict per epoch so
    channels stay grouped within a trial (needed for the per-trial ERP image).
    Each subject is loaded once; every candidate match in the session is cropped.
    """
    matches = [m for m in predictions[f"{session}.mp3"]
               if int(m["cluster_id"]) in cluster_ids
               and m["start_time"] < eeg.LISTEN_DURATIONS[session]]
    epochs = {cid: [] for cid in cluster_ids}
    if not matches:
        return epochs

    for subject in eeg.SUBJECTS:
        loaded = eeg.load_subject_epochs(subject, session)
        if loaded is None:
            continue
        present, crop_at = loaded
        for m in matches:
            data = crop_at(m["start_time"] + eeg.ERP_LAG)
            if data is None:
                continue
            epochs[int(m["cluster_id"])].append(dict(zip(present, data)))
    return epochs


def _channel_rows(epoch_dicts):
    """{channel: [epoch rows]} in the dataset's fixed temporal-channel order."""
    chan_rows = defaultdict(list)
    for ep in epoch_dicts:
        for ch, row in ep.items():
            chan_rows[ch].append(row)
    channels = [ch for ch in eeg.VERTEX_CHANNELS if ch in chan_rows]
    return channels, chan_rows


def _times(n):
    return eeg.ERP_LAG + np.arange(n) / eeg.RESAMPLE_HZ


def _title(session, cid, stats, n_epochs, extra):
    delta = stats["rms_r"] - stats["null_mean"]
    return (
        f"{session}  cluster {cid}   ({extra}, n_epochs={n_epochs})\n"
        f"rms_r={stats['rms_r']:.3f}   null={stats['null_mean']:.3f}   "
        f"Δ above null={delta:+.3f}   p95={stats['null_p95']:.3f}   p={stats['p']:.3g}"
    )


def _plot_heatmaps(session, cid, stats, epoch_dicts, out_dir):
    channels, chan_rows = _channel_rows(epoch_dicts)
    if not channels:
        return None

    # top: per-channel event-average, (C, T) in microvolts
    t_avg = min(len(r) for rows in chan_rows.values() for r in rows)
    avg = np.stack([
        np.stack([r[:t_avg] for r in chan_rows[ch]]).mean(axis=0)
        for ch in channels
    ]) * 1e6

    # bottom: per-epoch RMS across that epoch's channels, (N, T) in microvolts
    t_erp = min(len(next(iter(ep.values()))) for ep in epoch_dicts)
    erp = np.stack([
        np.sqrt(((np.stack([v[:t_erp] for v in ep.values()]) * 1e6) ** 2).mean(axis=0))
        for ep in epoch_dicts
    ])

    times_avg, times_erp = _times(avg.shape[1]), _times(erp.shape[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    vmax = max(np.percentile(np.abs(avg), 99), 1e-9)
    im1 = ax1.imshow(avg, aspect="auto", origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax,
                     extent=[times_avg[0], times_avg[-1], 0, len(channels)])
    ax1.set_ylabel("channel  (vertex cluster)")
    ax1.set_title(_title(session, cid, stats, len(epoch_dicts), f"{len(channels)} ch"))
    fig.colorbar(im1, ax=ax1, label="µV")

    vmax2 = max(np.percentile(erp, 99), 1e-9)
    im2 = ax2.imshow(erp, aspect="auto", origin="lower", cmap="magma",
                     vmin=0, vmax=vmax2,
                     extent=[times_erp[0], times_erp[-1], 0, erp.shape[0]])
    ax2.set_ylabel("epoch")
    ax2.set_xlabel("time since cluster onset (s)")
    fig.colorbar(im2, ax=ax2, label="RMS µV")

    fig.tight_layout()
    path = os.path.join(out_dir, f"{session}_cluster{cid}_heatmaps.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_sem(session, cid, stats, epoch_dicts, out_dir):
    channels, chan_rows = _channel_rows(epoch_dicts)
    if not channels:
        return None
    T = min(len(r) for rows in chan_rows.values() for r in rows)
    times = _times(T)

    fig, ax = plt.subplots(figsize=(9, 4))
    for ch in channels:
        m = np.stack([r[:T] for r in chan_rows[ch]]) * 1e6  # (n, T)
        mean = m.mean(axis=0)
        sem = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0]) if m.shape[0] > 1 else np.zeros(T)
        line, = ax.plot(times, mean, lw=0.6, alpha=0.7)
        ax.fill_between(times, mean - sem, mean + sem, color=line.get_color(), alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time since cluster onset (s)")
    ax.set_ylabel("µV  (mean ± SEM across epochs)")
    ax.set_title(_title(session, cid, stats, len(epoch_dicts), f"{len(channels)} ch"))

    fig.tight_layout()
    path = os.path.join(out_dir, f"{session}_cluster{cid}_sem.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def visualize(model_name, p_threshold=P_THRESHOLD):
    base = f"correlates/ds003774/{model_name}"
    with open(f"{base}/predictions.json") as f:
        predictions = json.load(f)
    with open(f"{base}/null_pvalues.json") as f:
        null_pvalues = json.load(f)

    candidates = _candidates(null_pvalues, p_threshold)
    n_total = sum(len(v) for v in candidates.values())
    if not n_total:
        print(f"No clusters cleared p < {p_threshold} for {model_name}.")
        return

    out_dir = f"{base}/evoked"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{n_total} candidate cluster(s) across {len(candidates)} session(s) -> {out_dir}/")

    with tqdm(total=n_total, desc="Rendering evoked responses") as pbar:
        for session, cluster_stats in candidates.items():
            epochs = _collect_epochs(session, set(cluster_stats), predictions)
            for cid, stats in cluster_stats.items():
                _plot_heatmaps(session, cid, stats, epochs[cid], out_dir)
                _plot_sem(session, cid, stats, epochs[cid], out_dir)
                pbar.write(f"  {session} cluster {cid}  "
                           f"(rms_r={stats['rms_r']:.3f}, p={stats['p']:.3g}, "
                           f"n={len(epochs[cid])})")
                pbar.update(1)


if __name__ == "__main__":
    model = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else P_THRESHOLD
    visualize(model, thresh)
