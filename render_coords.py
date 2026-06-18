"""Render an EEG electrode montage from a coords TSV (name, x, y, z).

Axes convention inferred from the ds003774 net (EGI/HydroCel GSN-129):
    +x = anterior (toward nose), y = left<->right, +z = up.
Mirror symmetry across the y=0 plane confirms y is the lateral axis.

Outputs a 3D scatter + a top-down 2D projection (nose up), labelled.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    names, xyz = [], []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names.append(row["name"])
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return names, np.asarray(xyz)


def render(path, out, label_every=1):
    names, xyz = load(path)
    x, y, z = xyz.T

    fig = plt.figure(figsize=(15, 7))

    # --- 3D view ---
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(x, y, z, c=z, cmap="viridis", s=25, depthshade=True)
    ax.set_xlabel("x  (anterior +)")
    ax.set_ylabel("y  (left <-> right)")
    ax.set_zlabel("z  (up +)")
    ax.set_title(f"3D montage — {len(names)} electrodes")
    # nose direction arrow
    ax.quiver(x.max(), 0, z.min(), 2.5, 0, 0, color="red", arrow_length_ratio=0.4)
    ax.text(x.max() + 2.8, 0, z.min(), "nose", color="red")
    try:
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    except Exception:
        pass

    # --- top-down 2D projection (looking down the +z axis, nose up) ---
    # screen vertical = x (anterior up); screen horizontal = y.
    # flip y so the subject's left hemisphere is on the LEFT of the image.
    ax2 = fig.add_subplot(1, 2, 2)
    hpos, vpos = -y, x
    sc = ax2.scatter(hpos, vpos, c=z, cmap="viridis", s=120, edgecolors="k", linewidths=0.4)
    for i, n in enumerate(names):
        if i % label_every == 0:
            ax2.annotate(n, (hpos[i], vpos[i]), fontsize=5, ha="center", va="center")

    # head outline: circle + nose triangle + ears
    r = max(np.abs(hpos).max(), np.abs(vpos).max()) * 1.08
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(r * np.cos(th), r * np.sin(th), "k-", lw=1)
    ax2.plot([-r * 0.12, 0, r * 0.12], [r, r * 1.12, r], "k-", lw=1)  # nose
    ax2.plot([-r, -r * 1.06, -r], [r * 0.12, 0, -r * 0.12], "k-", lw=1)  # left ear
    ax2.plot([r, r * 1.06, r], [r * 0.12, 0, -r * 0.12], "k-", lw=1)  # right ear
    ax2.text(0, -r * 1.05, "subject LEFT on image-left  |  nose up", ha="center", fontsize=8)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Top-down projection (z color = height)")
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, label="z (up)")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(names)} electrodes)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("coords", nargs="?", default="3774_coords.tsv")
    p.add_argument("-o", "--out", default="3774_coords.png")
    p.add_argument("--label-every", type=int, default=1, help="label every Nth electrode")
    a = p.parse_args()
    render(a.coords, a.out, a.label_every)
