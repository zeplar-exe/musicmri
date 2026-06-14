import argparse
import json
import os
from collections import defaultdict
from glob import glob

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
PCA_N_COMPONENTS = 200
CHUNK_FRAMES = int(librosa.time_to_frames(CHUNK_SIZE, sr=SR, hop_length=HOP_LENGTH))
CHUNK_DURATION = librosa.frames_to_time(CHUNK_FRAMES, sr=SR, hop_length=HOP_LENGTH) # true seconds/chunk (~0.998)
N_MELS = 128
PAD_DB = -80.0 # log-mel floor (power_to_db ref=np.max, top_db=80) (for padding short chunks)
SEED = 42


def is_junk(path):
    return os.path.basename(path).startswith("._") or "__MACOSX" in path


def chunk_audio(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    n_frames = log_mel.shape[1]
    if n_frames == 0:
        return np.empty((0, N_MELS * CHUNK_FRAMES))

    split_indices = np.arange(CHUNK_FRAMES, n_frames, CHUNK_FRAMES)
    chunks = np.split(log_mel, split_indices, axis=1)

    flattened = []
    
    for chunk in chunks:
        if chunk.shape[1] < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant", constant_values=PAD_DB)
        flattened.append(chunk.flatten())

    return np.array(flattened)


def load_and_chunk(path, offset=0.0, duration=None):
    audio, _ = librosa.load(path, sr=SR, offset=offset, duration=duration)
    return chunk_audio(audio)


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


def train(data_dir, model_dir, limit=None, per_chunk=False, per_band=False):
    keras.utils.set_random_seed(SEED) # seed for UMAP encoder
    embedder = ParametricUMAP(n_components=50, n_neighbors=15, random_state=SEED)

    samples = []
    sample_map = [] # which file each chunk came from
    time_map = [] # chunk start time within its own file

    wav_files = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    mp3_files = sorted(glob(os.path.join(data_dir, "**/*.mp3"), recursive=True))
    files = wav_files + mp3_files
    if limit is not None:
        files = files[:limit]

    for file in tqdm(files, desc="Loading audio files"):
        if is_junk(file):
            continue
        
        try:
            chunked = load_and_chunk(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            chunked = []
        
        if len(chunked) == 0:
            continue
        
        samples.extend(chunked)
        sample_map.extend([file] * len(chunked))
        time_map.extend([j * CHUNK_DURATION for j in range(len(chunked))])

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

    for size in range(5, 51, 5):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, prediction_data=True)
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

    n_clusters = len(set(best_labels[best_labels != -1]))
    print(f"Best silhouette: {best_score:.4f} with {n_clusters} clusters")

    os.makedirs(model_dir, exist_ok=True)
    embedder.encoder.save(os.path.join(model_dir, "umap_encoder.keras"))
    joblib.dump(pca, os.path.join(model_dir, "pca.joblib"))
    joblib.dump(best_model, os.path.join(model_dir, "hdbscan_model.joblib"))
    joblib.dump(norm, os.path.join(model_dir, "normalizer.joblib"))

    _dump_clusters(model_dir, sample_map, time_map, best_labels, best_model.probabilities_)



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
    """Soft membership: (n_chunks, n_clusters) probability over the real clusters.
    Column j is cluster id j; rows need not sum to 1 and argmax may differ from
    approximate_predict's hard label."""
    log_mels = _apply_norm(log_mels, norm)
    embedding = encoder.predict(pca.transform(log_mels).astype(np.float32), verbose=0)
    return np.atleast_2d(hdbscan.membership_vector(clusterer, embedding))


def _dump_clusters(model_dir, files, starts, labels, strengths):
    clusters = defaultdict(list)
    for file, start, label, strength in zip(files, starts, labels, strengths):
        clusters[int(label)].append({
            "file": file,
            "start": start,
            "end": start + CHUNK_DURATION,
            "strength": float(strength),
        })

    with open(os.path.join(model_dir, "clusters.json"), "w") as f:
        json.dump(clusters, f, indent=2)


def regenerate(data_dir, model_dir):
    """Rebuild clusters.json from the saved models, without retraining."""
    encoder, pca, clusterer, norm = _load_models(model_dir)

    wav_files = sorted(f for f in glob(os.path.join(data_dir, "**/*.wav"), recursive=True) if not is_junk(f))
    mp3_files = sorted(f for f in glob(os.path.join(data_dir, "**/*.mp3"), recursive=True) if not is_junk(f))
    files = wav_files + mp3_files

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


def exemplars(cluster_id, model_dir, n=10, out_dir="exemplars"):
    """Save the n highest-membership-strength audio chunks of one cluster."""
    with open(os.path.join(model_dir, "clusters.json")) as f:
        clusters = json.load(f)

    records = clusters.get(str(cluster_id))
    if not records:
        print(f"No chunks were assigned to cluster {cluster_id}.")
        return

    top = sorted(records, key=lambda r: r["strength"], reverse=True)[:n]
    gap = np.zeros(int(0.25 * SR), dtype=np.float32)

    clips = []
    for r in top:
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

    print(f"Wrote top {len(top)} of {len(records)} exemplars -> {out_path}")


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
    # scale each feature 0-9 relative to its min/max across real clusters (noise -1 excluded)
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


def profile(model_dir):
    """Average acoustic descriptors within each cluster"""
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

    p_ex = sub.add_parser("exemplars", help="Save the top-n-strength audio snippets of one cluster.")
    p_ex.add_argument("cluster", type=int)
    p_ex.add_argument("--model", dest="model_dir", required=True)
    p_ex.add_argument("--n", type=int, default=10)
    p_ex.add_argument("--out_dir", default="exemplars")

    p_prof = sub.add_parser("profile", help="Mean of named acoustic descriptors per cluster.")
    p_prof.add_argument("--model", dest="model_dir", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, args.model_dir, args.limit, args.per_chunk, args.per_band)
    elif args.command == "regenerate":
        regenerate(args.data_dir, args.model_dir)
    elif args.command == "predict":
        predict(args.mus_file, args.model_dir, args.start, args.end, args.soft, args.top_k)
    elif args.command == "exemplars":
        exemplars(args.cluster, args.model_dir, args.n, args.out_dir)
    elif args.command == "profile":
        profile(args.model_dir)
