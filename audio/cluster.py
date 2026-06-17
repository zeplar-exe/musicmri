import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from glob import glob

# cut out dead warnings
warnings.filterwarnings("ignore", message=r"`build\(\)` was called on layer 'umap_model'")

import librosa
import soundfile as sf
import numpy as np
import joblib
import hdbscan
import keras
from umap.parametric_umap import ParametricUMAP
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

SR = 22050 # to my knowledge, every .wav under inspection has sr=22050, whereas .mp3s need downsampling
CHUNK_SIZE = 1.0 # seconds
HOP_LENGTH = 512
PCA_N_COMPONENTS = 500
CHUNK_FRAMES = int(librosa.time_to_frames(CHUNK_SIZE, sr=SR, hop_length=HOP_LENGTH))
CHUNK_DURATION = librosa.frames_to_time(CHUNK_FRAMES, sr=SR, hop_length=HOP_LENGTH) # true seconds/chunk (~0.998)
N_MELS = 128
SEED = 42


def is_junk(path):
    return os.path.basename(path).startswith("._") or "__MACOSX" in path


def chunk_audio(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Keep only whole 1s chunks; drop the trailing partial one to avoid sub-1 second chunks
    n_chunks = log_mel.shape[1] // CHUNK_FRAMES
    if n_chunks == 0:
        return np.empty((0, N_MELS * CHUNK_FRAMES))

    flattened = [log_mel[:, k * CHUNK_FRAMES:(k + 1) * CHUNK_FRAMES].flatten()
                 for k in range(n_chunks)]
    return np.array(flattened)


def load_and_chunk(path, offset=0.0, duration=None):
    audio, _ = librosa.load(path, sr=SR, offset=offset, duration=duration)
    return chunk_audio(audio)


def chunk_rms_db(audio):
    """Per-chunk loudness for the energy gate: the MAX per-frame RMS within each
    chunk, in absolute dBFS (ref=1.0, NOT per-file). Absolute reference is what
    lets the gate tell a present instrument from a silent demucs residual.
    Aligned to chunk_audio's chunk grid, so index j matches one-to-one."""
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]  # per-frame, linear
    n_chunks = len(rms) // CHUNK_FRAMES  # whole chunks only, matching chunk_audio
    out = []
    for k in range(n_chunks):
        seg = rms[k * CHUNK_FRAMES:(k + 1) * CHUNK_FRAMES]
        out.append(20.0 * np.log10(float(seg.max()) + 1e-10))
    return np.array(out)


def _chunks_with_rms(path):
    """chunk_audio + chunk_rms_db from a single load (train-side energy gate)."""
    audio, _ = librosa.load(path, sr=SR)
    return chunk_audio(audio), chunk_rms_db(audio)


def _fit_norm(X, per_chunk, per_band):
    """Derive normalization params from the training samples (X: n x N_MELS*CHUNK_FRAMES)."""
    norm = {"per_chunk": per_chunk, "per_band": per_band, "band_mean": None, "band_std": None}

    if per_chunk: # loudness lives as a per-chunk dB offset; center it out before fitting bands
        X = X - X.mean(axis=1, keepdims=True)

    if per_band:
        bands = X.reshape(-1, N_MELS, CHUNK_FRAMES)
        norm["band_mean"] = bands.mean(axis=(0, 2)).astype(np.float32)
        norm["band_std"] = (bands.std(axis=(0, 2)) + 1e-8).astype(np.float32)

    return norm


def _apply_norm(X, norm):
    if norm["per_chunk"]:
        X = X - X.mean(axis=1, keepdims=True)

    if norm["per_band"]:
        bands = X.reshape(-1, N_MELS, CHUNK_FRAMES)
        bands = (bands - norm["band_mean"][None, :, None]) / norm["band_std"][None, :, None]
        X = bands.reshape(X.shape[0], -1)

    return X.astype(np.float32)


def _write_summary(model_dir, score, labels, extra=[]):
    labels = [int(x) for x in labels]
    total = len(labels)
    n_noise = sum(1 for x in labels if x == -1)
    sizes = defaultdict(int)
    
    for x in labels:
        if x != -1:
            sizes[x] += 1
    
    vals = sorted(sizes.values())
    n_clusters = len(vals)
    mn = vals[0] if vals else 0
    med = vals[len(vals) // 2] if vals else 0
    mx = vals[-1] if vals else 0
    
    lines = [
        f"Silhouette score: {score:.4f}",
        f"Clusters found: {n_clusters}",
        f"Noise points: {n_noise}/{total} ({n_noise / total:.1%})",
        f"Cluster sizes: min {mn}, med {med}, max {mx}",
    ]
    
    for e in extra:
        if e is not None:
            lines.append(e)

    with open(os.path.join(model_dir, "summary.txt"), "w") as f:
        for line in lines:
            if line is not None:
                f.write(line + "\n")

    print(" | ".join(line for line in lines if line is not None))


def _write_distributions(model_dir, files, labels):
    """Per-file and per-cluster breakdown."""
    per_file = defaultdict(lambda: defaultdict(int))
    for file, label in zip(files, labels):
        bn = os.path.basename(file)
        if int(label) != -1:
            per_file[bn][int(label)] += 1
    
    with open(os.path.join(model_dir, "distributions.txt"), "w") as f:
        for name in sorted(per_file):
            f.write(f"File {name}:\n")
            for cluster, count in sorted(per_file[name].items()):
                f.write(f"  Cluster {cluster}: {count} chunks\n")
    
    per_cluster = defaultdict(lambda: defaultdict(int))
    for file, label in zip(files, labels):
        bn = os.path.basename(file)
        if int(label) != -1:
            per_cluster[int(label)][bn] += 1
    
    with open(os.path.join(model_dir, "distributions-inverted.txt"), "w") as f:
        for name in sorted(per_cluster):
            f.write(f"Cluster {name}:\n")
            for file, count in sorted(per_cluster[name].items()):
                f.write(f"  File {file}: {count} chunks\n")


def train(data_dir, model_dir, limit=None, per_chunk=False, per_band=False,
          mcs_min=5, mcs_max=50, mcs_step=5, min_samples=None, min_rms=None):
    keras.utils.set_random_seed(SEED) # seed for UMAP encoder
    embedder = ParametricUMAP(n_components=50, n_neighbors=15, random_state=SEED)

    samples = []
    sample_map = [] # which file each chunk came from
    time_map = [] # chunk start time within its own file

    wav_files = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    mp3_files = sorted(glob(os.path.join(data_dir, "**/*.mp3"), recursive=True))
    flac_files = sorted(glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
    files = wav_files + mp3_files + flac_files
    if limit is not None:
        files = files[:limit]

    seen_chunks = 0  # total chunks before the energy gate, for the drop log
    for file in tqdm(files, desc="Loading audio files"):
        if is_junk(file):
            continue

        try:
            chunked, rms_db = _chunks_with_rms(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            chunked, rms_db = [], []

        if len(chunked) == 0:
            continue

        # Energy gate, drop near-silent chunks
        for j, chunk in enumerate(chunked):
            seen_chunks += 1
            if min_rms is not None and rms_db[j] < min_rms:
                continue
            samples.append(chunk)
            sample_map.append(file)
            time_map.append(j * CHUNK_DURATION)

    gate_note = None

    if min_rms is not None:
        dropped = seen_chunks - len(samples)
        gate_note = f"Gate dropped {dropped}/{seen_chunks} chunks < {min_rms:g} dBFS"
        print(f"Energy gate: {gate_note}")

    min_needed = max(mcs_max, min_samples or 0)
    
    if len(samples) < min_needed:
        reason = f"only {len(samples)}/{seen_chunks} chunks survived"
        if min_rms is not None:
            reason += f" the {min_rms:g} dBFS energy gate (stem likely silent)"
        raise RuntimeError(
            f"Too few chunks to cluster ({reason}, need >= {min_needed}); "
            f"skipping {data_dir}.")

    samples = np.asarray(samples, dtype=np.float32)
    print(f"Total samples: {len(samples)}")

    norm = _fit_norm(samples, per_chunk, per_band)
    samples = _apply_norm(samples, norm)
    active = [nm for nm, on in (("per-chunk", per_chunk), ("per-band", per_band)) if on]
    print(f"Normalization: {', '.join(active) if active else 'none'}")

    n_comp = min(PCA_N_COMPONENTS, samples.shape[0], samples.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    reduced = pca.fit_transform(samples).astype(np.float32)
    
    print(f"PCA: {n_comp} components retain {pca.explained_variance_ratio_.sum():.2%} of variance")

    embedding = embedder.fit_transform(reduced)

    best_score = -1
    best_labels = None
    best_model = None
    size = 0

    for size in range(mcs_min, mcs_max + 1, mcs_step):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=min_samples, prediction_data=True)
        labels = clusterer.fit_predict(embedding)

        mask = labels != -1
        
        if mask.sum() < 2 or len(set(labels[mask])) < 2: # sillhouete only defined for n > 2
            continue

        score = silhouette_score(embedding[mask], labels[mask], sample_size=min(2000, int(mask.sum())), random_state=SEED)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_model = clusterer

    if best_model is None:
        raise RuntimeError("No valid clustering found: every min_cluster_size produced <2 clusters.")

    os.makedirs(model_dir, exist_ok=True)
    embedder.encoder.save(os.path.join(model_dir, "umap_encoder.keras"))
    joblib.dump(pca, os.path.join(model_dir, "pca.joblib"))
    joblib.dump(best_model, os.path.join(model_dir, "hdbscan_model.joblib"))
    joblib.dump(norm, os.path.join(model_dir, "normalizer.joblib"))
    np.savez_compressed(os.path.join(model_dir, "embedding.npz"), embedding=embedding.astype(np.float32), labels=np.asarray(best_labels, dtype=np.int32))
    
    samples_note = f"Total samples: {len(samples)} ({CHUNK_SIZE:.2f}s chunks)"
    size_note = f"Best min_cluster_size: {size} (silhouette score {best_score:.4f})"

    _dump_clusters(model_dir, sample_map, time_map, best_labels, best_model.probabilities_)
    _write_summary(model_dir, best_score, best_labels, extra=[gate_note, samples_note, size_note])
    _write_distributions(model_dir, sample_map, best_labels)


def _load_models(model_dir):
    encoder = keras.models.load_model(os.path.join(model_dir, "umap_encoder.keras"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    clusterer = joblib.load(os.path.join(model_dir, "hdbscan_model.joblib"))

    norm_path = os.path.join(model_dir, "normalizer.joblib")
    norm = (joblib.load(norm_path) if os.path.exists(norm_path)
            else {"per_chunk": False, "per_band": False, "band_mean": None, "band_std": None})

    return encoder, pca, clusterer, norm


def _predict_chunks(log_mels, encoder, pca, clusterer, norm):
    log_mels = _apply_norm(log_mels, norm)
    embedding = encoder.predict(pca.transform(log_mels).astype(np.float32), verbose=0)
    return hdbscan.approximate_predict(clusterer, embedding)


def _predict_soft(log_mels, encoder, pca, clusterer, norm):
    """Probability cluster membership."""
    log_mels = _apply_norm(log_mels, norm)
    embedding = encoder.predict(pca.transform(log_mels).astype(np.float32), verbose=0)
    return np.atleast_2d(hdbscan.membership_vector(clusterer, embedding))


def _dump_clusters(model_dir, files, starts, labels, strengths):
    clusters = defaultdict(list)
    for file, start, label, strength in zip(files, starts, labels, strengths):
        clusters[int(label)].append({
            "file": file,
            "start": round(float(start), 4),
            "end": round(float(start + CHUNK_DURATION), 4),
            "strength": round(float(strength), 4),
        })

    ordered = {str(label): sorted(recs, key=lambda r: r["strength"], reverse=True) for label, recs in sorted(clusters.items())}

    with open(os.path.join(model_dir, "clusters.json"), "w") as f:
        json.dump(ordered, f, indent=2)


def regenerate(data_dir, model_dir):
    """Rebuild clusters.json from the saved models, without retraining."""
    encoder, pca, clusterer, norm = _load_models(model_dir)

    wav_files = sorted(f for f in glob(os.path.join(data_dir, "**/*.wav"), recursive=True) if not is_junk(f))
    mp3_files = sorted(f for f in glob(os.path.join(data_dir, "**/*.mp3"), recursive=True) if not is_junk(f))
    flac_files = sorted(f for f in glob(os.path.join(data_dir, "**/*.flac"), recursive=True) if not is_junk(f))
    files = wav_files + mp3_files + flac_files

    sample_map, time_map, all_labels, all_strengths = [], [], [], []
    for file in tqdm(files, desc="Predicting chunks"):
        try:
            log_mels = load_and_chunk(file)
        except Exception as e:
            print(f"Error loading {file}, skipping: {e}")
            continue

        if len(log_mels) == 0:
            continue

        labels, strengths = _predict_chunks(log_mels, encoder, pca, clusterer, norm)
        sample_map.extend([file] * len(labels))
        time_map.extend([j * CHUNK_DURATION for j in range(len(labels))])
        all_labels.extend(labels)
        all_strengths.extend(strengths)

    _dump_clusters(model_dir, sample_map, time_map, all_labels, all_strengths)
    print(f"Wrote clusters.json with {len(all_labels)} chunks.")


def predict(mus_file, model_dir, start_time=0.0, end_time=None, soft=False, top_k=3):
    duration = None if end_time is None else end_time - start_time
    samples = load_and_chunk(mus_file, offset=start_time, duration=duration)

    if len(samples) == 0:
        print("No audio to cluster.")
        return []

    encoder, pca, clusterer, norm = _load_models(model_dir)

    if soft:
        probs = _predict_soft(samples, encoder, pca, clusterer, norm)
        results = []
        for i, row in enumerate(probs):
            ts = start_time + i * CHUNK_DURATION
            order = np.argsort(row)[::-1][:top_k]
            top = "  ".join(f"c{j}:{row[j]:.2f}" for j in order)
            results.append((ts, row))
            print(f"{ts:.2f}-{ts + CHUNK_DURATION:.2f}s -> {top}")
        return results

    labels, strengths = _predict_chunks(samples, encoder, pca, clusterer, norm)

    results = []
    for i, label in enumerate(labels):
        ts = start_time + i * CHUNK_DURATION
        results.append((ts, int(label)))
        print(f"{ts:.2f}-{ts + CHUNK_DURATION:.2f}s -> cluster {label} (strength {strengths[i]:.2f})")

    return results


def _write_exemplar(records, cluster_id, n, out_dir):
    """Concatenate a cluster's exemplar (top 5 + random 5 interleaved) chunks into one wav."""
    if not records:
        print(f"No chunks were assigned to cluster {cluster_id}.")
        return

    half = n // 2
    ordered = sorted(records, key=lambda r: r["strength"], reverse=True)
    top = ordered[:half]

    rest = ordered[half:]
    rng = np.random.default_rng(SEED)
    n_rand = min(half, len(rest))
    rand = [rest[i] for i in rng.choice(len(rest), size=n_rand, replace=False)] if rest else []

    # interleave top/random, falling back to whichever list still has entries
    picks = []
    for i in range(max(len(top), len(rand))):
        if i < len(top):
            picks.append(top[i])
        if i < len(rand):
            picks.append(rand[i])

    gap = np.zeros(int(0.25 * SR), dtype=np.float32)
    clips = []
    for r in picks:
        seg, _ = librosa.load(r["file"], sr=SR, offset=r["start"], duration=CHUNK_DURATION)
        if len(seg) > 0:
            clips.append(seg)
            clips.append(gap)

    if not clips:
        print(f"Cluster {cluster_id} chunks all reloaded to empty audio.")
        return

    out = np.concatenate(clips)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cluster_{cluster_id}.wav")
    sf.write(out_path, out, SR)

    print(f"Wrote {len(top)} top + {len(rand)} random of {len(records)} exemplars -> {out_path}")


def exemplars(cluster_id, model_dir, n=10, out_dir="exemplars"):
    """Save the n highest-strength audio chunks of one cluster, or of every
    cluster (noise excluded) when cluster_id is None."""
    with open(os.path.join(model_dir, "clusters.json")) as f:
        clusters = json.load(f)

    if cluster_id is None:
        for key in sorted(clusters, key=int):
            if int(key) == -1:  # noise isn't a signature; skip it
                continue
            _write_exemplar(clusters[key], int(key), n, out_dir)
        return

    _write_exemplar(clusters.get(str(cluster_id)), cluster_id, n, out_dir)


DESCRIPTOR_NAMES = ["rms_db", "centroid", "bandwidth", "flux", "onset_rate"]
DESCRIPTOR_LETTERS = "lcbfo"  # Loudness, Centroid, Bandwidth, Flux, Onset-rate


def chunk_descriptors(y):
    rms = librosa.feature.rms(y=y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0].mean()
    flux = librosa.onset.onset_strength(y=y, sr=SR).mean()
    onset_rate = len(librosa.onset.onset_detect(y=y, sr=SR, units="time")) / CHUNK_DURATION

    return {
        "rms_db": float(librosa.amplitude_to_db(np.array([rms + 1e-8]))[0]),
        "centroid": float(centroid),
        "bandwidth": float(bandwidth),
        "flux": float(flux),
        "onset_rate": float(onset_rate),
    }


def cluster_means(model_dir):
    """Mean acoustic descriptors per cluster. Returns (labels, means, counts)."""
    with open(os.path.join(model_dir, "clusters.json")) as f:
        clusters = json.load(f)

    chunk_samples = int(CHUNK_DURATION * SR)

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    by_file = defaultdict(list)  # file -> [(label, start)]
    for label, records in clusters.items():
        for r in records:
            by_file[r["file"]].append((int(label), r["start"]))

    for file, items in tqdm(by_file.items(), desc="Profiling clusters"):
        try:
            audio, _ = librosa.load(file, sr=SR)
        except Exception as e:
            print(f"Error loading {file}, skipping: {e}")
            continue

        for label, start in items:
            i = int(round(start * SR))
            seg = audio[i:i + chunk_samples]

            if len(seg) < chunk_samples // 2:
                continue

            for k, v in chunk_descriptors(seg).items():
                sums[label][k] += v

            counts[label] += 1

    labels = sorted(counts)
    means = {lab: {nm: sums[lab][nm] / counts[lab] for nm in DESCRIPTOR_NAMES} for lab in labels}
    return labels, means, dict(counts)


def name_clusters(labels, means):
    """Build the LCBFO label for each cluster (digit = 0-9 position between min/max)."""
    real = [lab for lab in labels if lab != -1] or labels
    lo = {nm: min(means[lab][nm] for lab in real) for nm in DESCRIPTOR_NAMES}
    hi = {nm: max(means[lab][nm] for lab in real) for nm in DESCRIPTOR_NAMES}

    def cluster_name(m):
        out = []
        for letter, nm in zip(DESCRIPTOR_LETTERS, DESCRIPTOR_NAMES):
            span = hi[nm] - lo[nm]
            d = 0 if span == 0 else round((m[nm] - lo[nm]) / span * 9)
            out.append(f"{letter}{max(0, min(9, d))}")
        return "".join(out)

    return {lab: cluster_name(means[lab]) for lab in labels}


def _plot_heatmap(labels, means, counts, out_dir):
    """Clusters x 5 descriptors, z-scored so colors are comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = DESCRIPTOR_NAMES
    cluster_name = name_clusters(labels, means)
    rows = [lab for lab in labels if lab != -1] or labels

    M = np.array([[means[lab][nm] for nm in names] for lab in rows], dtype=float)
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True) + 1e-8
    Z = (M - mu) / sd

    fig_h = max(3.0, 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    im = ax.imshow(Z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab} {cluster_name[lab]} (n={counts[lab]})" for lab in rows], fontsize=6)
    ax.set_title("Cluster acoustic profiles (z-scored per feature)")
    fig.colorbar(im, ax=ax, label="std devs from mean")
    fig.tight_layout()

    path = os.path.join(out_dir, "cluster_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


def _plot_scatter(model_dir, out_dir, max_chunks=20000):
    """2D scatter of 50-D -> PCA 2-D embedding, colored by cluster."""
    emb_path = os.path.join(model_dir, "embedding.npz")
    if not os.path.exists(emb_path):
        print("No embedding.npz (retrain to enable the scatter); skipping.", file=sys.stderr)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(emb_path)
    emb, labels = data["embedding"], data["labels"]

    if len(emb) > max_chunks:
        idx = np.random.default_rng(SEED).choice(len(emb), max_chunks, replace=False)
        emb, labels = emb[idx], labels[idx]

    xy = PCA(n_components=2, random_state=SEED).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 7))

    noise = labels == -1
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=3, c="lightgray", alpha=0.4, label="noise (-1)")

    real = sorted(set(labels[~noise].tolist()))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(real):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i % 20), alpha=0.6)
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.text(cx, cy, str(lab), fontsize=7, weight="bold", ha="center", va="center")

    ax.set_title(f"Embedding (PCA of 50-D UMAP) - {len(real)} clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    path = os.path.join(out_dir, "cluster_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


def profile(model_dir, out_dir=None):
    """Average acoustic descriptors within each cluster. Draws heatmap + scatter PNGs."""
    labels, means, counts = cluster_means(model_dir)
    names = DESCRIPTOR_NAMES
    cluster_name = name_clusters(labels, means)

    header = (f"{'cluster':>8} {'n':>7} " + " ".join(f"{nm:>11}" for nm in names) + f"  {'name':>12}")

    print(header)
    print("-" * len(header))

    for lab in labels:
        c = counts[lab]
        row = " ".join(f"{means[lab][nm]:>11.2f}" for nm in names)
        print(f"{lab:>8} {c:>7} {row}  {cluster_name[lab]:>12}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _plot_heatmap(labels, means, counts, out_dir)
        _plot_scatter(model_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised clustering of audio chunks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fit the UMAP embedder + HDBSCAN over a data dir.")
    p_train.add_argument("--data", dest="data_dir", required=True)
    p_train.add_argument("--model", dest="model_dir", required=True)
    p_train.add_argument("--limit", type=int, default=None,
                         help="Only load the first N files.")
    p_train.add_argument("--per-chunk", action="store_true",
                         help="Per-chunk loudness normalization (remove each chunk's dB level).")
    p_train.add_argument("--per-band", action="store_true",
                         help="Per-band z-score across the corpus (remove average spectral shape).")
    p_train.add_argument("--mcs-min", type=int, default=5,
                         help="Smallest HDBSCAN min_cluster_size to try (default 5).")
    p_train.add_argument("--mcs-max", type=int, default=50,
                         help="Largest HDBSCAN min_cluster_size to try (default 50).")
    p_train.add_argument("--mcs-step", type=int, default=5,
                         help="Step between min_cluster_size values in the sweep (default 5).")
    p_train.add_argument("--min-samples", type=int, default=None,
                         help="HDBSCAN min_samples: the noise/conservatism dial. Higher = more "
                              "points become noise. Default: tied to each min_cluster_size.")
    p_train.add_argument("--min-rms", type=float, default=None,
                         help="Energy gate (dBFS): drop chunks whose max per-frame RMS is below "
                              "this before clustering. Absolute reference. Default off; try -50 "
                              "for stems to skip silence/absent-instrument residual.")

    p_regen = sub.add_parser("regenerate", help="Rebuild clusters.json from saved models without retraining.")
    p_regen.add_argument("--data", dest="data_dir",  required=True)
    p_regen.add_argument("--model", dest="model_dir", required=True)

    p_pred = sub.add_parser("predict", help="Cluster every chunk of one audio file.")
    p_pred.add_argument("mus_file")
    p_pred.add_argument("--model", dest="model_dir", required=True)
    p_pred.add_argument("--start", type=float, default=0.0)
    p_pred.add_argument("--end", type=float, default=None)
    p_pred.add_argument("--soft", action="store_true",
                        help="Print the soft membership probability vector instead of one hard label.")
    p_pred.add_argument("--top-k", dest="top_k", type=int, default=3,
                        help="With --soft, how many top clusters to show per chunk.")

    p_ex = sub.add_parser("exemplars", help="Save the top-n-strength audio snippets of one cluster (or all).")
    p_ex.add_argument("cluster", type=int, nargs="?", default=None,
                      help="Cluster id. Omit to write exemplars for every cluster (noise excluded).")
    p_ex.add_argument("--model", dest="model_dir", required=True)
    p_ex.add_argument("--n", type=int, default=10)
    p_ex.add_argument("--out", dest="out_dir", default="exemplars")

    p_prof = sub.add_parser("profile", help="Mean of named acoustic descriptors per cluster.")
    p_prof.add_argument("--model", dest="model_dir", required=True)
    p_prof.add_argument("--out-dir", dest="out_dir", default=None,
                        help="If set, also write cluster_heatmap.png + cluster_scatter.png here.")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, args.model_dir, args.limit, args.per_chunk, args.per_band,
              args.mcs_min, args.mcs_max, args.mcs_step, args.min_samples, args.min_rms)
    elif args.command == "regenerate":
        regenerate(args.data_dir, args.model_dir)
    elif args.command == "predict":
        predict(args.mus_file, args.model_dir, args.start, args.end, args.soft, args.top_k)
    elif args.command == "exemplars":
        exemplars(args.cluster, args.model_dir, args.n, args.out_dir)
    elif args.command == "profile":
        profile(args.model_dir, args.out_dir)
