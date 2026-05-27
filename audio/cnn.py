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
    # Dynamics
    "crescendo", "decrescendo", "sforzando", "swell",
    "fortepiano", "subito_forte", "subito_piano", "morendo",
    # Articulation
    "staccato", "legato", "marcato", "tenuto", "portato", "accent",
    # Ornaments
    "vibrato", "tremolo", "trill", "glissando", "mordent",
    "grace_note", "pitch_bend", "turn", "portamento",
    # Tempo modification
    "accelerando", "ritardando", "rubato", "fermata", "caesura",
    # Timbre modification
    "muted", "harmonics", "wah", "distortion", "palm_mute",
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

        self.samples = []
        for wav_path in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            json_path = os.path.join(label_dir, f"{stem}.json")
            if os.path.exists(json_path):
                self.samples.append((wav_path, json_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No matched audio/label pairs found in {data_dir}. "
                f"Expected audio/*.wav and labels/*.json with matching names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, json_path = self.samples[idx]

        # Features
        mel = self.fe(wav_path)  # (n_mels, n_frames)
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

    @staticmethod
    def _pad_or_truncate(arr, target_frames):
        """Pad or truncate along the last axis."""
        n = arr.shape[-1]
        if n >= target_frames:
            return arr[..., :target_frames]
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_frames - n)]
        return np.pad(arr, pad_width, mode="constant", constant_values=0)


def collate_fn(batch):
    """Collate variable-length spectrograms by padding to the longest in the batch."""
    mels, labels = zip(*batch)
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

    return torch.stack(padded_mels), torch.stack(padded_labels)


# ============================================================
# Model
# ============================================================

class TechniqueCNN(nn.Module):
    """Frame-level multi-label classifier.

    Architecture:
        4 conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool on freq axis only)
        Collapse frequency dimension
        1x1 Conv1d to per-frame label predictions
        Sigmoid output

    Input:  (batch, 1, n_mels, n_frames)
    Output: (batch, num_labels, n_frames)  -- sigmoid activations
    """

    def __init__(self, n_mels=128, num_labels=NUM_LABELS):
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

        # Per-frame classification via 1x1 convolution over time
        self.classifier = nn.Sequential(
            nn.Conv1d(collapsed_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, num_labels, kernel_size=1),
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

    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_mels, n_frames)
        Returns:
            (batch, num_labels, n_frames) sigmoid activations
        """
        x = self.conv_blocks(x)             # (batch, 256, freq', n_frames)
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)          # (batch, 2048, n_frames)
        x = self.classifier(x)              # (batch, num_labels, n_frames)
        return torch.sigmoid(x)


# ============================================================
# Training
# ============================================================

def train(data_dir, epochs=50, batch_size=16, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fe = FeatureExtractor()
    dataset = TechniqueDataset(data_dir, fe)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"Loaded {len(dataset)} samples")

    model = TechniqueCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for mel_batch, label_batch in tqdm(loader, desc=f"Epoch {epoch}"):
            mel_batch = mel_batch.to(device)
            label_batch = label_batch.to(device)

            pred = model(mel_batch)  # (batch, num_labels, n_frames)
            loss = criterion(pred, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    # Save model
    save_path = os.path.join(data_dir, "model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "labels": LABELS,
        "fe_params": {
            "sr": fe.sr, "n_fft": fe.n_fft,
            "hop_length": fe.hop_length, "n_mels": fe.n_mels,
        },
    }, save_path)
    print(f"Saved model to {save_path}")


def finetune(model_path, data_dir, epochs=20, batch_size=8, lr=1e-4,
             save_path=None, device=None):
    """Load existing model and fine-tune on new data.

    Args:
        model_path: path to existing model.pt
        data_dir:   directory with audio/ and labels/ subdirs (corrected data)
        epochs:     fine-tuning epochs (fewer than from-scratch)
        batch_size: smaller batch for small datasets
        lr:         lower learning rate to avoid catastrophic forgetting
        save_path:  where to save the updated model (default: overwrite model_path)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning on {device}")

    # Load existing model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    fe_params = checkpoint["fe_params"]

    fe = FeatureExtractor(**fe_params)
    model = TechniqueCNN(n_mels=fe_params["n_mels"], num_labels=len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    # Load new data
    dataset = TechniqueDataset(data_dir, fe)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"Fine-tuning on {len(dataset)} corrected samples")

    # Lower LR, only train classifier head initially if dataset is very small
    if len(dataset) < 20:
        print("Small dataset: freezing conv blocks, training classifier only")
        for param in model.conv_blocks.parameters():
            param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = model.parameters()

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for mel_batch, label_batch in loader:
            mel_batch = mel_batch.to(device)
            label_batch = label_batch.to(device)

            pred = model(mel_batch)
            loss = criterion(pred, label_batch)

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
    }, out)
    print(f"Saved fine-tuned model to {out}")
    return out


# ============================================================
# Inference
# ============================================================

MIN_DURATION = {
    # Dynamics
    "crescendo": 0.4,
    "decrescendo": 0.4,
    "sforzando": 0.05,
    "swell": 0.5,
    "fortepiano": 0.08,
    "subito_forte": 0.1,
    "subito_piano": 0.1,
    "morendo": 0.4,
    # Articulation
    "staccato": 0.3,
    "legato": 0.3,
    "marcato": 0.2,
    "tenuto": 0.2,
    "portato": 0.2,
    "accent": 0.08,
    # Ornaments
    "vibrato": 0.2,
    "tremolo": 0.15,
    "trill": 0.1,
    "glissando": 0.1,
    "mordent": 0.04,
    "grace_note": 0.03,
    "pitch_bend": 0.1,
    "turn": 0.08,
    "portamento": 0.1,
    # Tempo modification
    "accelerando": 0.5,
    "ritardando": 0.5,
    "rubato": 0.5,
    "fermata": 0.3,
    "caesura": 0.15,
    # Timbre modification
    "muted": 0.15,
    "harmonics": 0.1,
    "wah": 0.2,
    "distortion": 0.15,
    "palm_mute": 0.1,
}


def predict(audio_path, model_path, threshold=0.5, device=None):
    """Run inference on a single audio file.

    Returns:
        List of {"label": str, "start": float, "end": float, "confidence": float}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    fe_params = checkpoint["fe_params"]
    labels = checkpoint["labels"]

    fe = FeatureExtractor(**fe_params)
    model = TechniqueCNN(n_mels=fe_params["n_mels"], num_labels=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mel = fe(audio_path)                          # (n_mels, n_frames)
    mel_tensor = torch.from_numpy(mel[np.newaxis, np.newaxis, :, :]).to(device)

    with torch.no_grad():
        pred = model(mel_tensor)                  # (1, num_labels, n_frames)
    pred = pred.squeeze(0).cpu().numpy()           # (num_labels, n_frames)

    # Convert frame-level predictions to spans
    results = []
    n_frames = pred.shape[1]
    frame_duration = fe.hop_length / fe.sr  # seconds per frame

    for li, label in enumerate(labels):
        active = pred[li] >= threshold
        # Find contiguous runs of True
        spans = _contiguous_spans(active)
        for start_frame, end_frame in spans:
            confidence = float(pred[li, start_frame:end_frame].mean())
            results.append({
                "label": label,
                "start": round(start_frame * frame_duration, 3),
                "end": round(end_frame * frame_duration, 3),
                "confidence": round(confidence, 3),
            })

    results = [r for r in results
               if r["end"] - r["start"] >= MIN_DURATION.get(r["label"], 0.15)]
    results.sort(key=lambda r: r["start"])
    return results


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

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--audio", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--threshold", type=float, default=0.5)

    p_ft = sub.add_parser("finetune")
    p_ft.add_argument("--model", required=True, help="Existing model.pt to fine-tune")
    p_ft.add_argument("--data_dir", required=True, help="Corrected data (audio/ + labels/)")
    p_ft.add_argument("--epochs", type=int, default=20)
    p_ft.add_argument("--batch_size", type=int, default=8)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--save", default=None, help="Output path (default: overwrite)")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
    elif args.command == "predict":
        results = predict(args.audio, args.model, threshold=args.threshold)
        print(json.dumps(results, indent=2))
    elif args.command == "finetune":
        finetune(args.model, args.data_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, save_path=args.save)
    else:
        parser.print_help()