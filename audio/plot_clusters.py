"""Visualize a trained cluster.py model: a clusters x features heatmap (what each
cluster *is*) and a 2D embedding scatter (how well the clusters separate).

Both read from a saved model dir. The heatmap also needs clusters.json (already
there after train/regenerate). The scatter re-extracts chunks from a data dir,
runs them through the saved encoder, and squashes the 50-D embedding to 2-D.

> Thanks Claude.
"""

import argparse
import os
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import cluster


def plot_heatmap(model_dir, out_dir):
    """Clusters x 5 descriptors, each column z-scored so colors are comparable."""
    labels, means, counts = cluster.cluster_means(model_dir)
    names = cluster.DESCRIPTOR_NAMES
    cluster_name = cluster.name_clusters(labels, means)

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
    print(f"Wrote {path}")


def _embed_corpus(model_dir, data_dir, max_chunks=20000):
    """Re-extract chunks from data_dir, embed them with the saved encoder, and
    return (embedding_50d, labels). Subsamples to max_chunks for a readable plot."""
    encoder, pca, clusterer, norm = cluster._load_models(model_dir)

    wav = sorted(f for f in glob(os.path.join(data_dir, "**/*.wav"), recursive=True) if not cluster.is_junk(f))
    mp3 = sorted(f for f in glob(os.path.join(data_dir, "**/*.mp3"), recursive=True) if not cluster.is_junk(f))
    files = wav + mp3

    chunks = []
    for file in tqdm(files, desc="Loading audio"):
        try:
            c = cluster.load_and_chunk(file)
        except Exception as e:
            print(f"Error loading {file}, skipping: {e}")
            continue
        if len(c):
            chunks.append(c)

    if not chunks:
        raise RuntimeError(f"No audio chunks found under {data_dir}.")

    X = np.concatenate(chunks).astype(np.float32)

    if len(X) > max_chunks:
        idx = np.random.default_rng(cluster.SEED).choice(len(X), max_chunks, replace=False)
        X = X[idx]

    labels, _ = cluster._predict_chunks(X, encoder, pca, clusterer, norm)

    Xn = cluster._apply_norm(X, norm)
    emb = encoder.predict(pca.transform(Xn).astype(np.float32), verbose=0)
    return emb, np.asarray(labels)


def plot_scatter(model_dir, data_dir, out_dir, max_chunks=20000):
    """2D scatter of the embedding (50-D -> PCA 2-D), colored by cluster."""
    emb, labels = _embed_corpus(model_dir, data_dir, max_chunks)

    xy = PCA(n_components=2, random_state=cluster.SEED).fit_transform(emb)

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
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a trained cluster.py model.")
    parser.add_argument("--model", dest="model_dir", required=True)
    parser.add_argument("--data", dest="data_dir", default=None,
                        help="Data dir to embed for the scatter. If omitted, only the heatmap is drawn.")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--max-chunks", type=int, default=20000,
                        help="Subsample this many chunks for the scatter.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_heatmap(args.model_dir, args.out_dir)
    if args.data_dir:
        plot_scatter(args.model_dir, args.data_dir, args.out_dir, args.max_chunks)
    else:
        print("No --data given; skipped the scatter (heatmap only).")
