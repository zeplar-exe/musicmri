"""
Frame-level multi-label CNN for detecting musical expressive techniques.

Input:  audio waveform (mono, any sample rate)
Output: per-frame binary predictions across all technique labels

Usage:
    # Training
    python technique_cnn.py train --data_dir ./data --epochs 50

    # Inference
    python technique_cnn.py predict --audio my_file.wav --model model.pt

Data format:
    data_dir/
        audio/
            001.wav
            002.wav
        labels/
            001.json    # [{"label": "crescendo", "start": 0.5, "end": 2.1}, ...]
            002.json
"""

import json
import os
import glob
from tqdm import tqdm
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Label definitions
# ============================================================

LABELS = [
    # Amplitude dynamics
    "crescendo", "decrescendo", "swell", "subito_piano", "accent",
    # Articulation
    "staccato", "legato",
    # Pitch movement
    "vibrato", "trill", "glissando", "grace_note", "pitch_bend",
    "scoop", "fall_off",
    # Tempo / rhythm
    "accelerando", "ritardando", "fermata", "caesura",
    # Percussion / texture
    "roll", "choke",
    # Pitch pattern
    "arpeggio",
]

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


# ============================================================
# Feature extraction
# ============================================================

class FeatureExtractor:
    """Compute mel spectrogram from audio."""

    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio_path):
        """Load audio and return log-mel spectrogram as numpy array.

        Returns:
            mel: (n_mels, n_frames) float32 array
        """
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
        )
        # Log scale, clamp to avoid log(0)
        mel = np.log(np.maximum(mel, 1e-10))
        return mel.astype(np.float32)

    def seconds_to_frame(self, seconds):
        """Convert a time in seconds to the nearest spectrogram frame index."""
        return int(round(seconds * self.sr / self.hop_length))


# ============================================================
# Dataset
# ============================================================

class TechniqueDataset(Dataset):
    """Loads audio + label pairs, returns mel spectrograms and frame-level
    binary label matrices.

    Label files are JSON: [{"label": "vibrato", "start": 0.5, "end": 2.0}, ...]
    """

    def __init__(self, data_dir, feature_extractor, max_frames=None):
        self.fe = feature_extractor
        self.max_frames = max_frames  # truncate/pad to fixed length if set

        audio_dir = os.path.join(data_dir, "audio")
        label_dir = os.path.join(data_dir, "labels")
        cache_dir = os.path.join(data_dir, "mel_cache")

        self.samples = []
        for wav_path in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            json_path = os.path.join(label_dir, f"{stem}.json")
            if os.path.exists(json_path):
                cache_path = os.path.join(cache_dir, f"{stem}.npy")
                self.samples.append((wav_path, json_path, cache_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No matched audio/label pairs found in {data_dir}. "
                f"Expected audio/*.wav and labels/*.json with matching names."
            )

        # Precompute mel cache if missing or stale
        uncached = [s for s in self.samples
                    if not os.path.exists(s[2])
                    or os.path.getmtime(s[0]) > os.path.getmtime(s[2])]
        if uncached:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching {len(uncached)} mel spectrograms...")
            for i, (wav_path, _, cache_path) in enumerate(uncached):
                mel = self.fe(wav_path)
                np.save(cache_path, mel)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(uncached)}")
            print(f"  Cached to {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, json_path, cache_path = self.samples[idx]

        mel = np.load(cache_path)  # (n_mels, n_frames)
        n_frames = mel.shape[1]

        # Labels: frame-level binary matrix (num_labels, n_frames)
        label_matrix = np.zeros((NUM_LABELS, n_frames), dtype=np.float32)
        with open(json_path) as f:
            annotations = json.load(f)
        for ann in annotations:
            label_name = ann["label"]
            if label_name not in LABEL_TO_IDX:
                continue
            li = LABEL_TO_IDX[label_name]
            start_frame = self.fe.seconds_to_frame(ann["start"])
            end_frame = self.fe.seconds_to_frame(ann["end"])
            start_frame = max(0, start_frame)
            end_frame = min(n_frames, end_frame)
            label_matrix[li, start_frame:end_frame] = 1.0

        # Optional: pad/truncate to fixed length
        if self.max_frames is not None:
            mel = self._pad_or_truncate(mel, self.max_frames)
            label_matrix = self._pad_or_truncate(label_matrix, self.max_frames)

        # Add channel dim for conv2d: (1, n_mels, n_frames)
        mel = mel[np.newaxis, :, :]

        return torch.from_numpy(mel), torch.from_numpy(label_matrix)

    def label_frame_stats(self):
        """Count positive frames per label and total frames across the set.

        Reads label JSON + the cached mel header only (mmap, no data load),
        so this is cheap even for tens of thousands of samples.

        Returns (pos_frames: np.ndarray[NUM_LABELS], total_frames: int).
        """
        pos = np.zeros(NUM_LABELS, dtype=np.float64)
        total = 0
        for _, json_path, cache_path in self.samples:
            n_frames = np.load(cache_path, mmap_mode="r").shape[1]
            total += n_frames
            with open(json_path) as f:
                annotations = json.load(f)
            for ann in annotations:
                if ann["label"] not in LABEL_TO_IDX:
                    continue
                li = LABEL_TO_IDX[ann["label"]]
                s = max(0, self.fe.seconds_to_frame(ann["start"]))
                e = min(n_frames, self.fe.seconds_to_frame(ann["end"]))
                pos[li] += max(0, e - s)
        return pos, total

    @staticmethod
    def _pad_or_truncate(arr, target_frames):
        """Pad or truncate along the last axis."""
        n = arr.shape[-1]
        if n >= target_frames:
            return arr[..., :target_frames]
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_frames - n)]
        return np.pad(arr, pad_width, mode="constant", constant_values=0)


def collate_fn(batch):
    """Collate variable-length spectrograms by padding to the longest in the batch.

    Returns (mels, labels, lengths) where lengths holds each sample's true
    frame count so the GRU can pack and the loss can mask padded frames.
    """
    mels, labels = zip(*batch)
    lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
    max_t = max(m.shape[-1] for m in mels)

    padded_mels = []
    padded_labels = []
    for m, l in zip(mels, labels):
        pad_t = max_t - m.shape[-1]
        if pad_t > 0:
            m = torch.nn.functional.pad(m, (0, pad_t))
            l = torch.nn.functional.pad(l, (0, pad_t))
        padded_mels.append(m)
        padded_labels.append(l)

    return torch.stack(padded_mels), torch.stack(padded_labels), lengths


def make_frame_mask(lengths, n_frames, device):
    """(batch, 1, n_frames) float mask — 1.0 on real frames, 0.0 on padding."""
    idx = torch.arange(n_frames, device=device)[None, :]
    mask = (idx < lengths.to(device)[:, None]).float()
    return mask.unsqueeze(1)


def spec_augment(mel, n_freq=2, freq_w=15, n_time=2, time_w=20):
    """SpecAugment: mask random freq bands + time steps to the per-sample mean.

    Applied during training only. Masking to the mean (not 0) is correct for
    log-mel, whose silence floor is strongly negative.
    """
    b, _, F, T = mel.shape
    out = mel.clone()
    for i in range(b):
        fill = mel[i].mean()
        for _ in range(n_freq):
            w = int(torch.randint(0, freq_w + 1, (1,)))
            if w:
                f0 = int(torch.randint(0, max(1, F - w), (1,)))
                out[i, :, f0:f0 + w, :] = fill
        for _ in range(n_time):
            w = int(torch.randint(0, time_w + 1, (1,)))
            if w:
                t0 = int(torch.randint(0, max(1, T - w), (1,)))
                out[i, :, :, t0:t0 + w] = fill
    return out


def masked_bce(logits, targets, mask, criterion):
    """Per-frame BCE over real frames only.

    criterion: BCEWithLogitsLoss(reduction='none', pos_weight=...)
    mask:      (batch, 1, n_frames) from make_frame_mask
    """
    loss_map = criterion(logits, targets) * mask     # broadcast over labels
    denom = mask.sum() * targets.shape[1]            # frames * num_labels
    return loss_map.sum() / denom.clamp(min=1)


# ============================================================
# Model
# ============================================================

class TechniqueCNN(nn.Module):
    """Frame-level multi-label classifier.

    Architecture:
        4 conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool on freq axis only)
        Collapse frequency dimension
        1x1 Conv1d projection
        Bidirectional GRU over time (full-sequence temporal context)
        1x1 Conv1d to per-frame label logits

    Input:  (batch, 1, n_mels, n_frames)
    Output: (batch, num_labels, n_frames)  -- raw logits (apply sigmoid downstream)
    """

    def __init__(self, n_mels=128, num_labels=NUM_LABELS, proj_dim=256, gru_hidden=256):
        super().__init__()

        # Each block: conv2d -> bn -> relu -> pool(freq only)
        # Pool kernel (2,1) halves frequency, preserves time
        self.conv_blocks = nn.Sequential(
            self._block(1,   32, pool=(2, 1)),   # n_mels: 128 -> 64
            self._block(32,  64, pool=(2, 1)),   # 64 -> 32
            self._block(64, 128, pool=(2, 1)),   # 32 -> 16
            self._block(128, 256, pool=(2, 1)),  # 16 -> 8
        )

        # After conv blocks: (batch, 256, 8, n_frames)
        # Collapse frequency: reshape to (batch, 256*8, n_frames)
        collapsed_dim = 256 * (n_mels // 16)  # 256 * 8 = 2048

        # Project collapsed freq features down before temporal modeling
        self.proj = nn.Sequential(
            nn.Conv1d(collapsed_dim, proj_dim, kernel_size=1),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )

        # BiGRU gives each frame context from the whole sequence — needed for
        # long techniques (crescendo, accel/rit, fermata) that exceed the conv
        # stack's ~0.4s time receptive field.
        self.gru = nn.GRU(proj_dim, gru_hidden, num_layers=2, dropout=0.2,
                          batch_first=True, bidirectional=True)

        # Per-frame classification via 1x1 convolution over time -> logits
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(2 * gru_hidden, num_labels, kernel_size=1),
        )

    @staticmethod
    def _block(in_ch, out_ch, pool=(2, 1)):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x:       (batch, 1, n_mels, n_frames)
            lengths: optional (batch,) true frame counts for packing the GRU
                     over variable-length, zero-padded inputs.
        Returns:
            (batch, num_labels, n_frames) raw logits
        """
        x = self.conv_blocks(x)             # (batch, 256, freq', n_frames)
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)          # (batch, 2048, n_frames)
        x = self.proj(x)                    # (batch, proj_dim, n_frames)
        x = x.transpose(1, 2)               # (batch, n_frames, proj_dim)
        if lengths is not None:
            # Pack so the backward GRU pass never sees padding (time dim is
            # preserved through the conv stack, so frame lengths stay valid).
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed, _ = self.gru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed, batch_first=True, total_length=t)
        else:
            x, _ = self.gru(x)              # (batch, n_frames, 2*gru_hidden)
        x = x.transpose(1, 2)               # (batch, 2*gru_hidden, n_frames)
        x = self.classifier(x)              # (batch, num_labels, n_frames) logits
        return x


# ============================================================
# Training
# ============================================================

def compute_pos_weight(dataset, clamp_max=50.0, device="cpu"):
    """Per-label positive-class weight = neg_frames / pos_frames, clamped.

    Counters the extreme negative dominance of frame-level labels so rare,
    short techniques aren't drowned out by easy-negative frames.
    """
    pos, total = dataset.label_frame_stats()
    neg = total - pos
    w = np.clip(neg / np.maximum(pos, 1.0), 1.0, clamp_max)
    return torch.tensor(w, dtype=torch.float32).view(1, NUM_LABELS, 1).to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_labels):
    """Validation pass: masked loss + per-label precision/recall/F1 at 0.5."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp = torch.zeros(num_labels)
    fp = torch.zeros(num_labels)
    fn = torch.zeros(num_labels)

    for mel_batch, label_batch, lengths in loader:
        mel_batch = mel_batch.to(device)
        label_batch = label_batch.to(device)
        logits = model(mel_batch, lengths)
        mask = make_frame_mask(lengths, label_batch.shape[-1], device)
        total_loss += masked_bce(logits, label_batch, mask, criterion).item()
        n_batches += 1

        pred = ((torch.sigmoid(logits) >= 0.5).float() * mask)
        tgt = label_batch * mask
        tp += (pred * tgt).sum(dim=(0, 2)).cpu()
        fp += (pred * (1 - tgt)).sum(dim=(0, 2)).cpu()
        fn += ((1 - pred) * tgt).sum(dim=(0, 2)).cpu()

    prec = tp / (tp + fp).clamp(min=1)
    rec = tp / (tp + fn).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
    return total_loss / max(n_batches, 1), f1.mean().item(), prec, rec, f1


@torch.no_grad()
def sweep_thresholds(model, loader, device, labels):
    """Pick the per-label threshold that maximizes F1 on the validation set."""
    model.eval()
    probs = [[] for _ in labels]
    tgts = [[] for _ in labels]

    for mel_batch, label_batch, lengths in loader:
        logits = model(mel_batch.to(device), lengths)
        prob = torch.sigmoid(logits).cpu()
        mask = make_frame_mask(lengths, label_batch.shape[-1], "cpu").bool()
        mask = mask.expand(-1, len(labels), -1)
        for li in range(len(labels)):
            sel = mask[:, li, :]
            probs[li].append(prob[:, li, :][sel])
            tgts[li].append(label_batch[:, li, :][sel])

    grid = torch.linspace(0.1, 0.9, 33)
    thresholds = {}
    for li, label in enumerate(labels):
        p = torch.cat(probs[li])
        t = torch.cat(tgts[li])
        if t.sum() == 0:
            thresholds[label] = 0.5
            continue
        best_f1, best_thr = -1.0, 0.5
        for thr in grid:
            pred = (p >= thr).float()
            tp = (pred * t).sum()
            fp = (pred * (1 - t)).sum()
            fn = ((1 - pred) * t).sum()
            prec = tp / (tp + fp).clamp(min=1)
            rec = tp / (tp + fn).clamp(min=1)
            f1 = (2 * prec * rec / (prec + rec).clamp(min=1e-8)).item()
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds[label] = round(best_thr, 3)
    return thresholds


def train(data_dir, epochs=50, batch_size=16, lr=1e-3, val_frac=0.1, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    fe = FeatureExtractor()
    dataset = TechniqueDataset(data_dir, fe)
    print(f"Loaded {len(dataset)} samples")

    # Positive-class weights from the full set (mild global prior)
    pos_weight = compute_pos_weight(dataset, device=device)

    # Deterministic train/val split
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    split_gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=split_gen)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)
    print(f"Train: {n_train}  Val: {n_val}")

    model = TechniqueCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for mel_batch, label_batch, lengths in tqdm(train_loader, desc=f"Epoch {epoch}"):
            mel_batch = spec_augment(mel_batch.to(device))
            label_batch = label_batch.to(device)

            logits = model(mel_batch, lengths)  # (batch, num_labels, n_frames)
            mask = make_frame_mask(lengths, label_batch.shape[-1], device)
            loss = masked_bce(logits, label_batch, mask, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_loss, macro_f1, _, _, _ = evaluate(model, val_loader, criterion, device, NUM_LABELS)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_macroF1={macro_f1:.3f}")

    # Restore best checkpoint before threshold sweep + save
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val macro-F1: {best_f1:.3f}")

    print("Sweeping per-label thresholds on val...")
    thresholds = sweep_thresholds(model, val_loader, device, LABELS)

    _, _, prec, rec, f1 = evaluate(model, val_loader, criterion, device, NUM_LABELS)
    print("\nPer-label validation metrics (thr=0.5):")
    print(f"  {'label':14s} {'prec':>5s} {'rec':>5s} {'f1':>5s}  {'best_thr':>8s}")
    for li, label in enumerate(LABELS):
        print(f"  {label:14s} {prec[li]:5.2f} {rec[li]:5.2f} {f1[li]:5.2f}  "
              f"{thresholds[label]:8.3f}")

    save_path = os.path.join(data_dir, "model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "labels": LABELS,
        "fe_params": {
            "sr": fe.sr, "n_fft": fe.n_fft,
            "hop_length": fe.hop_length, "n_mels": fe.n_mels,
        },
        "thresholds": thresholds,
    }, save_path)
    print(f"Saved model to {save_path}")


def finetune(model_path, data_dir, epochs=20, batch_size=8, lr=1e-4,
             base_data_dir=None, base_weight=0.5, save_path=None, device=None):
    """Load existing model and fine-tune on corrected data, optionally mixed
    with original training data to prevent catastrophic forgetting.

    Args:
        model_path:     path to existing model.pt
        data_dir:       directory with corrected audio/ and labels/
        epochs:         fine-tuning epochs
        batch_size:     batch size
        lr:             learning rate
        base_data_dir:  original synthetic training data dir; if provided,
                        a random subset is mixed in each epoch
        base_weight:    fraction of each batch that comes from base data (0-1).
                        e.g. 0.5 means half corrected, half synthetic.
        save_path:      where to save (default: overwrite model_path)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Fine-tuning on {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    fe_params = checkpoint["fe_params"]

    fe = FeatureExtractor(**fe_params)
    model = TechniqueCNN(n_mels=fe_params["n_mels"], num_labels=len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    # Load corrected data
    corrected_ds = TechniqueDataset(data_dir, fe)
    print(f"Corrected samples: {len(corrected_ds)}")

    # Load base data if provided
    base_ds = None
    if base_data_dir and os.path.isdir(os.path.join(base_data_dir, "audio")):
        base_ds = TechniqueDataset(base_data_dir, fe)
        print(f"Base (synthetic) samples: {len(base_ds)}")

        # Mix: oversample corrected data to match desired ratio
        # Each epoch sees all corrected samples + a proportional chunk of base
        n_base_per_epoch = int(len(corrected_ds) * base_weight / max(1 - base_weight, 0.1))
        n_base_per_epoch = min(n_base_per_epoch, len(base_ds))

        from torch.utils.data import ConcatDataset, Subset
        mixed_datasets = True
        print(f"Mixing {len(corrected_ds)} corrected + {n_base_per_epoch} base per epoch "
              f"({base_weight:.0%} base)")
    else:
        mixed_datasets = False
        if base_data_dir:
            print(f"Warning: base_data_dir '{base_data_dir}' not found, fine-tuning on corrections only")

    # Freeze conv blocks for very small corrected sets
    if len(corrected_ds) < 20:
        print("Small dataset: freezing conv blocks, training classifier only")
        for param in model.conv_blocks.parameters():
            param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = model.parameters()

    optimizer = torch.optim.Adam(trainable, lr=lr)
    # Positive-class weights from base data if mixed, else the corrections
    pos_weight = compute_pos_weight(base_ds or corrected_ds, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    for epoch in range(1, epochs + 1):
        model.train()

        if mixed_datasets:
            # Random subset of base data each epoch for variety
            base_indices = torch.randperm(len(base_ds))[:n_base_per_epoch].tolist()
            base_subset = Subset(base_ds, base_indices)
            epoch_ds = ConcatDataset([corrected_ds, base_subset])
        else:
            epoch_ds = corrected_ds

        loader = DataLoader(
            epoch_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )

        total_loss = 0
        n_batches = 0

        for mel_batch, label_batch, lengths in loader:
            mel_batch = spec_augment(mel_batch.to(device))
            label_batch = label_batch.to(device)

            logits = model(mel_batch, lengths)
            mask = make_frame_mask(lengths, label_batch.shape[-1], device)
            loss = masked_bce(logits, label_batch, mask, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    # Unfreeze everything for the save
    for param in model.parameters():
        param.requires_grad = True

    out = save_path or model_path
    torch.save({
        "model_state": model.state_dict(),
        "labels": checkpoint["labels"],
        "fe_params": fe_params,
        "thresholds": checkpoint.get("thresholds", {}),
    }, out)
    print(f"Saved fine-tuned model to {out}")
    return out


# ============================================================
# Inference
# ============================================================

MIN_DURATION = {
    # Amplitude dynamics
    "crescendo": 0.4,
    "decrescendo": 0.4,
    "swell": 0.5,
    "subito_piano": 0.1,
    "accent": 0.08,
    # Articulation
    "staccato": 0.3,
    "legato": 0.3,
    # Pitch movement
    "vibrato": 0.2,
    "trill": 0.1,
    "glissando": 0.1,
    "grace_note": 0.03,
    "pitch_bend": 0.1,
    "scoop": 0.05,
    "fall_off": 0.05,
    # Tempo / rhythm
    "accelerando": 0.5,
    "ritardando": 0.5,
    "fermata": 0.3,
    "caesura": 0.15,
    # Percussion / texture
    "roll": 0.2,
    "choke": 0.05,
    # Pitch pattern
    "arpeggio": 0.15,
}


def predict(audio_path, model_path, threshold=0.5, device=None):
    """Run inference on a single audio file.

    Returns:
        List of {"label": str, "start": float, "end": float, "confidence": float}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    fe_params = checkpoint["fe_params"]
    labels = checkpoint["labels"]
    label_thresholds = checkpoint.get("thresholds", {})

    fe = FeatureExtractor(**fe_params)
    model = TechniqueCNN(n_mels=fe_params["n_mels"], num_labels=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mel = fe(audio_path)                          # (n_mels, n_frames)
    mel_tensor = torch.from_numpy(mel[np.newaxis, np.newaxis, :, :]).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(mel_tensor))   # (1, num_labels, n_frames)
    pred = pred.squeeze(0).cpu().numpy()           # (num_labels, n_frames)

    # Convert frame-level predictions to spans
    results = []
    n_frames = pred.shape[1]
    frame_duration = fe.hop_length / fe.sr  # seconds per frame

    for li, label in enumerate(labels):
        thr = label_thresholds.get(label, threshold)
        active = pred[li] >= thr
        # Find contiguous runs of True
        spans = _contiguous_spans(active)
        for start_frame, end_frame in spans:
            confidence = float(pred[li, start_frame:end_frame].mean())
            span_scores = pred[:, start_frame:end_frame].mean(axis=1)
            top10_idx = np.argsort(span_scores)[::-1][:10]
            top10 = [{"label": labels[j], "score": round(float(span_scores[j]), 3)}
                     for j in top10_idx]
            results.append({
                "label": label,
                "start": round(start_frame * frame_duration, 3),
                "end": round(end_frame * frame_duration, 3),
                "confidence": round(confidence, 3),
                "top10": top10,
            })

    results = [r for r in results
               if r["end"] - r["start"] >= MIN_DURATION.get(r["label"], 0.15)]
    results.sort(key=lambda r: r["start"])
    return results


def separate_stems(audio_path, device=None):
    """Separate audio into 6 stems using Demucs (htdemucs_6s).

    Returns:
        dict mapping stem name -> path to separated WAV file.
        Stems: drums, bass, vocals, guitar, piano, other
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import tempfile
    import soundfile as sf

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # htdemucs_6s doesn't support MPS — fall back to CPU
    if device == "mps":
        device = "cpu"

    model = get_model("htdemucs_6s")
    model.to(device)

    # Load audio via librosa (handles mp3/wav/flac)
    sr = model.samplerate
    y, _ = librosa.load(audio_path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono -> stereo
    waveform = torch.from_numpy(y).float()  # (2, samples)

    # Apply model: returns (sources, channels, samples)
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    sources = apply_model(model, waveform[None].to(device), device=device)[0]
    sources = sources * ref.std() + ref.mean()

    # Save each stem to a temp directory
    stem_dir = tempfile.mkdtemp(prefix="demucs_stems_")
    stem_paths = {}

    for i, stem_name in enumerate(model.sources):
        stem_audio = sources[i].cpu().numpy()
        if stem_audio.ndim == 2:
            stem_audio = stem_audio.mean(axis=0)  # mono
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        sf.write(stem_path, stem_audio, sr)
        stem_paths[stem_name] = stem_path

    return stem_paths


def predict_stems(audio_path, model_path, threshold=0.5, device=None):
    """Separate audio into stems via Demucs, predict on each, merge results.

    Each prediction is tagged with a 'stem' field indicating which stem
    it came from. Duplicate detections across stems are deduplicated by
    keeping the highest-confidence one.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Separating stems...")
    stem_paths = separate_stems(audio_path, device=device)
    print(f"  Separated into {len(stem_paths)} stems: {list(stem_paths.keys())}")

    all_results = []
    for stem_name, stem_path in stem_paths.items():
        print(f"  Predicting on {stem_name}...")
        results = predict(stem_path, model_path, threshold=threshold, device=device)
        for r in results:
            r["stem"] = stem_name
        all_results.extend(results)
        print(f"    {len(results)} predictions")

    # Clean up temp files
    import shutil
    stem_dir = os.path.dirname(list(stem_paths.values())[0])
    shutil.rmtree(stem_dir, ignore_errors=True)

    # Deduplicate: if same label overlaps in time across stems, keep highest confidence
    all_results.sort(key=lambda r: (r["label"], r["start"]))
    deduped = []
    for r in all_results:
        merged = False
        for existing in deduped:
            if (existing["label"] == r["label"]
                    and r["start"] < existing["end"]
                    and r["end"] > existing["start"]):
                # Overlapping same label — keep higher confidence, widen span
                if r["confidence"] > existing["confidence"]:
                    existing["confidence"] = r["confidence"]
                    existing["top10"] = r["top10"]
                    existing["stem"] = r["stem"]
                existing["start"] = min(existing["start"], r["start"])
                existing["end"] = max(existing["end"], r["end"])
                merged = True
                break
        if not merged:
            deduped.append(r)

    deduped.sort(key=lambda r: r["start"])
    print(f"  Total after dedup: {len(deduped)} predictions")
    return deduped


def _contiguous_spans(bool_array):
    """Find contiguous True regions in a 1D boolean array.
    Returns list of (start_idx, end_idx) tuples."""
    spans = []
    in_span = False
    start = 0
    for i, v in enumerate(bool_array):
        if v and not in_span:
            start = i
            in_span = True
        elif not v and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(bool_array)))
    return spans


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Musical technique detector")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data_dir", required=True)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val_frac", type=float, default=0.1,
                         help="Fraction of data held out for validation")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--audio", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--threshold", type=float, default=0.5)
    p_pred.add_argument("--stems", action="store_true",
                        help="Separate into stems (Demucs 6s) before predicting")

    p_ft = sub.add_parser("finetune")
    p_ft.add_argument("--model", required=True, help="Existing model.pt to fine-tune")
    p_ft.add_argument("--data_dir", required=True, help="Corrected data (audio/ + labels/)")
    p_ft.add_argument("--base_data", default=None,
                      help="Original training data dir to mix in (prevents forgetting)")
    p_ft.add_argument("--base_weight", type=float, default=0.5,
                      help="Fraction of base data in each batch (0-1, default: 0.5)")
    p_ft.add_argument("--epochs", type=int, default=20)
    p_ft.add_argument("--batch_size", type=int, default=8)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--save", default=None, help="Output path (default: overwrite)")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac)
    elif args.command == "predict":
        if args.stems:
            results = predict_stems(args.audio, args.model, threshold=args.threshold)
        else:
            results = predict(args.audio, args.model, threshold=args.threshold)
        print(json.dumps(results, indent=2))
    elif args.command == "finetune":
        finetune(args.model, args.data_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr,
                 base_data_dir=args.base_data, base_weight=args.base_weight,
                 save_path=args.save)
    else:
        parser.print_help()