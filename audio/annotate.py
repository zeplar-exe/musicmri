"""
Prediction reviewer for musical technique detection.

FastAPI backend + HTML/JS frontend with WaveSurfer.js.

Pre-computes all audio segments and reference data URLs at startup,
so no heavy work happens during UI interaction.

Usage:
    python annotate.py --audio song.wav --model model.pt
    python annotate.py --audio song.wav --predictions preds.json
    python annotate.py --audio song.wav --predictions preds.json --ref_dir ./reference
"""

import argparse
import json
import os
import io
import base64
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from cnn import TechniqueCNN, FeatureExtractor, LABELS, MIN_DURATION


# ============================================================
# Audio helpers
# ============================================================

def load_audio(path, sr=22050):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def segment_data_url(y, sr, start_sec, end_sec, context_sec=0.5):
    s = max(0, int((start_sec - context_sec) * sr))
    e = min(len(y), int((end_sec + context_sec) * sr))
    segment = y[s:e]
    buf = io.BytesIO()
    sf.write(buf, segment, sr, format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:audio/wav;base64,{b64}"


def wav_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:audio/wav;base64,{b64}"


def find_ref_wav(ref_dir, label):
    if not ref_dir or not os.path.isdir(ref_dir):
        return None
    exact = os.path.join(ref_dir, f"{label}.wav")
    if os.path.isfile(exact):
        return exact
    for f in os.listdir(ref_dir):
        if f.endswith(".wav") and label in f.lower():
            return os.path.join(ref_dir, f)
    return None


# ============================================================
# Inference
# ============================================================

def run_inference(audio_path, model_path, threshold=0.3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    fe_params = checkpoint["fe_params"]
    labels = checkpoint["labels"]

    fe = FeatureExtractor(**fe_params)
    model = TechniqueCNN(n_mels=fe_params["n_mels"], num_labels=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mel = fe(audio_path)
    mel_tensor = torch.from_numpy(mel[np.newaxis, np.newaxis, :, :]).to(device)
    with torch.no_grad():
        pred = model(mel_tensor)
    pred = pred.squeeze(0).cpu().numpy()

    frame_dur = fe.hop_length / fe.sr
    predictions = []
    for li, label in enumerate(labels):
        active = pred[li] >= threshold
        in_span = False
        start = 0
        for fi, v in enumerate(active):
            if v and not in_span:
                start = fi
                in_span = True
            elif not v and in_span:
                conf = float(pred[li, start:fi].mean())
                predictions.append({
                    "label": label,
                    "start": round(start * frame_dur, 4),
                    "end": round(fi * frame_dur, 4),
                    "confidence": round(conf, 3),
                })
                in_span = False
        if in_span:
            conf = float(pred[li, start:].mean())
            predictions.append({
                "label": label,
                "start": round(start * frame_dur, 4),
                "end": round(pred.shape[1] * frame_dur, 4),
                "confidence": round(conf, 3),
            })

    predictions = [p for p in predictions
                   if p["end"] - p["start"] >= MIN_DURATION.get(p["label"], 0.15)]
    predictions.sort(key=lambda p: (p["start"], p["end"]))
    return predictions


# ============================================================
# Pre-computation
# ============================================================

def precompute(waveform, sr, predictions, ref_dir):
    """Build all data needed by the frontend, indexed by prediction index."""
    print("Pre-computing audio segments...")
    segments = []
    for i, pred in enumerate(predictions):
        segments.append(segment_data_url(waveform, sr, pred["start"], pred["end"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(predictions)} segments")
    print(f"  Done: {len(segments)} segments")

    print("Loading reference audio...")
    ref_urls = {}
    for label in LABELS:
        path = find_ref_wav(ref_dir, label)
        if path:
            ref_urls[label] = wav_to_data_url(path)
    print(f"  Loaded {len(ref_urls)} reference clips")

    return segments, ref_urls


# ============================================================
# State
# ============================================================

class ReviewState:
    def __init__(self, audio_path, predictions, out_path, ref_dir,
                 train_dir=None, model_path=None):
        self.audio_path = audio_path
        self.out_path = out_path
        self.ref_dir = ref_dir
        self.train_dir = train_dir
        self.model_path = model_path
        self.waveform, self.sr = load_audio(audio_path)
        self.duration = len(self.waveform) / self.sr
        self.predictions = predictions
        self.current_idx = 0
        self.decisions = {}  # idx -> {"action": "confirm"|"reject"|"relabel", "label": str}

    @property
    def total(self):
        return len(self.predictions)

    def confirm(self, idx, label=None):
        pred = self.predictions[idx]
        self.decisions[idx] = {
            "action": "relabel" if label and label != pred["label"] else "confirm",
            "label": label or pred["label"],
        }

    def reject(self, idx):
        self.decisions[idx] = {"action": "reject"}

    def clear_decision(self, idx):
        self.decisions.pop(idx, None)

    def get_accepted(self):
        accepted = []
        for idx, dec in sorted(self.decisions.items()):
            if dec["action"] in ("confirm", "relabel"):
                pred = self.predictions[idx]
                accepted.append({
                    "label": dec["label"],
                    "start": pred["start"],
                    "end": pred["end"],
                })
        return accepted

    def save(self):
        accepted = self.get_accepted()
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(accepted, f, indent=2)

        if self.train_dir and accepted:
            audio_dir = os.path.join(self.train_dir, "audio")
            label_dir = os.path.join(self.train_dir, "labels")
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            stem = Path(self.audio_path).stem
            dst_audio = os.path.join(audio_dir, f"{stem}.wav")
            dst_label = os.path.join(label_dir, f"{stem}.json")

            if not os.path.exists(dst_audio):
                if self.audio_path.endswith(".wav"):
                    shutil.copy2(self.audio_path, dst_audio)
                else:
                    sf.write(dst_audio, self.waveform, self.sr)

            with open(dst_label, "w") as f:
                json.dump(accepted, f, indent=2)

        return self.out_path

    def run_finetune(self, epochs=20, lr=1e-4):
        if not self.model_path or not self.train_dir:
            return None, "No model or train_dir set"

        from cnn import finetune

        audio_dir = os.path.join(self.train_dir, "audio")
        if not os.path.isdir(audio_dir):
            return None, f"No audio in {audio_dir}"
        n_files = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        if n_files == 0:
            return None, "No training files accumulated yet"

        save_path = self.model_path
        finetune(
            model_path=self.model_path,
            data_dir=self.train_dir,
            epochs=epochs,
            lr=lr,
            save_path=save_path,
        )
        return save_path, f"Fine-tuned on {n_files} files, saved to {save_path}"


# ============================================================
# App
# ============================================================

def create_app(state: ReviewState, segments, ref_urls):
    app = FastAPI()

    full_audio_url = wav_to_data_url(state.audio_path) if state.audio_path.endswith(".wav") else None
    if not full_audio_url:
        buf = io.BytesIO()
        sf.write(buf, state.waveform, state.sr, format="WAV")
        buf.seek(0)
        full_audio_url = f"data:audio/wav;base64,{base64.b64encode(buf.read()).decode()}"

    @app.get("/api/init")
    def api_init():
        return {
            "predictions": state.predictions,
            "labels": LABELS,
            "duration": state.duration,
            "total": state.total,
            "audioUrl": full_audio_url,
            "refUrls": ref_urls,
            "filename": os.path.basename(state.audio_path),
            "canRetrain": bool(state.model_path and state.train_dir),
        }

    @app.get("/api/segment/{idx}")
    def api_segment(idx: int):
        if 0 <= idx < len(segments):
            return {"url": segments[idx]}
        return JSONResponse({"error": "out of range"}, 404)

    @app.get("/api/decisions")
    def api_decisions():
        return {
            "decisions": {str(k): v for k, v in state.decisions.items()},
            "stats": {
                "confirmed": sum(1 for d in state.decisions.values() if d["action"] == "confirm"),
                "relabeled": sum(1 for d in state.decisions.values() if d["action"] == "relabel"),
                "rejected": sum(1 for d in state.decisions.values() if d["action"] == "reject"),
                "remaining": state.total - len(state.decisions),
            },
        }

    @app.post("/api/confirm/{idx}")
    def api_confirm(idx: int, label: str = None):
        state.confirm(idx, label)
        return {"ok": True}

    @app.post("/api/reject/{idx}")
    def api_reject(idx: int):
        state.reject(idx)
        return {"ok": True}

    @app.post("/api/clear/{idx}")
    def api_clear(idx: int):
        state.clear_decision(idx)
        return {"ok": True}

    @app.post("/api/save")
    def api_save():
        path = state.save()
        accepted = state.get_accepted()
        return {"path": path, "count": len(accepted)}

    @app.post("/api/retrain")
    def api_retrain():
        path, msg = state.run_finetune()
        return {"path": path, "message": msg, "ok": path is not None}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    return app


# ============================================================
# Frontend
# ============================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Technique Reviewer</title>
<script src="https://unpkg.com/wavesurfer.js@7"></script>
<script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
<style>
  :root {
    --bg: #111318;
    --card: #1a1d24;
    --border: #2a2d35;
    --border-hi: #4a9eff;
    --text: #e0e2e8;
    --dim: #6b7280;
    --green: #34d399;
    --red: #f87171;
    --orange: #fb923c;
    --blue: #60a5fa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }
  .container { max-width: 960px; margin: 0 auto; padding: 20px; }

  /* Header */
  .header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .file { color: var(--dim); font-size: 13px; }

  /* Stats bar */
  .stats {
    display: flex; gap: 20px; font-size: 13px; color: var(--dim);
    margin-bottom: 16px; padding: 8px 12px;
    background: var(--card); border-radius: 6px; border: 1px solid var(--border);
  }
  .stats .confirmed { color: var(--green); }
  .stats .relabeled { color: var(--blue); }
  .stats .rejected { color: var(--red); }

  /* Top 10 */
  .top10 {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; margin-bottom: 16px;
  }
  .top10 .label { color: var(--dim); font-size: 11px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .top10-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .top10-item {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 5px; font-size: 12px;
    background: var(--bg); border: 1px solid var(--border);
    cursor: pointer; transition: all 0.12s;
  }
  .top10-item:hover { border-color: var(--border-hi); color: var(--text); }
  .top10-item.active { border-color: var(--orange); background: #2a1f14; }
  .top10-item .t10-label { color: var(--text); }
  .top10-item .t10-conf { color: var(--dim); font-size: 11px; }
  .top10-item .t10-time { color: var(--dim); font-size: 10px; font-family: monospace; }

  /* Waveform */
  .waveform-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; margin-bottom: 16px;
  }
  .waveform-card .label { color: var(--dim); font-size: 11px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

  /* Prediction info */
  .pred-info {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  .pred-title {
    font-size: 16px; font-weight: 600; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px;
  }
  .pred-title .technique { color: var(--orange); }
  .pred-title .conf { color: var(--dim); font-size: 13px; font-weight: 400; }
  .pred-meta { color: var(--dim); font-size: 13px; }
  .pred-status { margin-top: 8px; font-size: 13px; font-weight: 500; }
  .pred-status.confirmed { color: var(--green); }
  .pred-status.relabeled { color: var(--blue); }
  .pred-status.rejected { color: var(--red); }

  /* Audio players */
  .audio-section {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  .audio-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
  }
  .audio-row:last-child { margin-bottom: 0; }
  .audio-label { color: var(--dim); font-size: 12px; min-width: 80px; text-transform: uppercase; letter-spacing: 0.5px; }
  .audio-row audio { height: 32px; flex: 1; }

  /* Relabel dropdown */
  .relabel-section {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  .relabel-section label { color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .relabel-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  select.relabel-select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 14px; flex: 1; max-width: 250px;
    cursor: pointer;
  }
  select.relabel-select:focus { border-color: var(--border-hi); outline: none; }
  .listen-btn {
    background: none; border: 1px solid var(--border); color: var(--dim);
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px;
    transition: all 0.15s;
  }
  .listen-btn:hover { border-color: var(--border-hi); color: var(--text); }

  /* Action buttons */
  .actions {
    display: flex; gap: 10px; justify-content: center;
    margin-bottom: 16px; flex-wrap: wrap;
  }
  .btn {
    border: none; border-radius: 6px; padding: 10px 28px; font-size: 14px;
    font-weight: 500; cursor: pointer; transition: all 0.15s;
  }
  .btn:hover { filter: brightness(1.15); }
  .btn-confirm { background: #065f46; color: var(--green); }
  .btn-reject { background: #7f1d1d; color: var(--red); }
  .btn-undo { background: var(--card); color: var(--dim); border: 1px solid var(--border); }
  .btn-undo:hover { color: var(--text); border-color: var(--border-hi); }

  /* Navigation */
  .nav {
    display: flex; gap: 10px; justify-content: center; align-items: center;
    margin-bottom: 16px;
  }
  .nav-btn {
    background: var(--card); border: 1px solid var(--border); color: var(--dim);
    border-radius: 6px; padding: 8px 20px; cursor: pointer; font-size: 13px;
    transition: all 0.15s;
  }
  .nav-btn:hover:not(:disabled) { border-color: var(--border-hi); color: var(--text); }
  .nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .nav-counter { color: var(--dim); font-size: 13px; min-width: 100px; text-align: center; }

  /* Prediction list (sidebar-like) */
  .pred-list {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; margin-bottom: 16px; max-height: 240px; overflow-y: auto;
  }
  .pred-list-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 8px; border-radius: 4px; cursor: pointer; font-size: 12px;
    transition: background 0.1s;
  }
  .pred-list-item:hover { background: #22252d; }
  .pred-list-item.active { background: #1e293b; border: 1px solid var(--border-hi); }
  .pred-list-item .lbl { color: var(--text); }
  .pred-list-item .time { color: var(--dim); font-family: monospace; font-size: 11px; }
  .pred-list-item .dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }
  .dot-pending { background: var(--dim); opacity: 0.3; }
  .dot-confirmed { background: var(--green); }
  .dot-relabeled { background: var(--blue); }
  .dot-rejected { background: var(--red); }

  /* Save / retrain section */
  .finish-section {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 16px; text-align: center;
  }
  .finish-section .btn {
    margin: 4px;
  }
  .btn-save { background: #1e40af; color: var(--blue); }
  .btn-retrain { background: #4a1d7a; color: #c084fc; }
  .finish-msg { color: var(--dim); font-size: 13px; margin-top: 10px; }

  /* Loading */
  .loading { text-align: center; padding: 60px; color: var(--dim); }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="container" id="app">
  <div class="loading">Loading...</div>
</div>

<script>
let D = {};       // init data
let idx = 0;      // current prediction index
let decisions = {};
let segmentCache = {};

async function init() {
  D = await (await fetch('/api/init')).json();
  decisions = {};
  await refreshDecisions();
  render();
  loadWaveform();
}

async function refreshDecisions() {
  const r = await (await fetch('/api/decisions')).json();
  decisions = r.decisions;
  return r.stats;
}

async function getSegment(i) {
  if (segmentCache[i]) return segmentCache[i];
  const r = await (await fetch(`/api/segment/${i}`)).json();
  segmentCache[i] = r.url;
  return r.url;
}

// ---- Waveform ----

let ws = null;
let wsRegions = null;

function loadWaveform() {
  const el = document.getElementById('waveform');
  if (!el) return;

  if (ws) { ws.destroy(); ws = null; }

  wsRegions = WaveSurfer.Regions.create();

  ws = WaveSurfer.create({
    container: el,
    waveColor: '#3b82f6',
    progressColor: '#1d4ed8',
    cursorColor: '#4a9eff',
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    height: 80,
    url: D.audioUrl,
    plugins: [wsRegions],
  });

  ws.on('ready', () => { updateRegions(); });
}

function updateRegions() {
  if (!ws || !wsRegions) return;
  wsRegions.clearRegions();

  D.predictions.forEach((p, i) => {
    const dec = decisions[String(i)];
    let color = 'rgba(107,114,128,0.15)';
    if (i === idx) color = 'rgba(251,146,60,0.35)';
    else if (dec) {
      if (dec.action === 'confirm') color = 'rgba(52,211,153,0.2)';
      else if (dec.action === 'relabel') color = 'rgba(96,165,250,0.2)';
      else if (dec.action === 'reject') color = 'rgba(248,113,113,0.15)';
    }
    wsRegions.addRegion({
      start: p.start, end: p.end, color,
      drag: false, resize: false,
    });
  });
}

// ---- Top 10 ----

function top10Html() {
  const ranked = D.predictions
    .map((p, i) => ({...p, idx: i}))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10);
  return ranked.map(p =>
    `<div class="top10-item ${p.idx === idx ? 'active' : ''}" onclick="jump(${p.idx})">
      <span class="t10-label">${p.label.replace(/_/g, ' ')}</span>
      <span class="t10-conf">${(p.confidence * 100).toFixed(0)}%</span>
      <span class="t10-time">${p.start.toFixed(1)}s</span>
    </div>`
  ).join('');
}

// ---- Rendering ----

function render() {
  const pred = D.predictions[idx];
  const dec = decisions[String(idx)];
  const stats = computeStats();

  document.getElementById('app').innerHTML = `
    <div class="header">
      <h1>Technique Reviewer</h1>
      <span class="file">${D.filename} &mdash; ${D.duration.toFixed(1)}s</span>
    </div>

    <div class="stats">
      <span>${D.total} predictions</span>
      <span class="confirmed">&#10003; ${stats.confirmed} confirmed</span>
      <span class="relabeled">&#9998; ${stats.relabeled} relabeled</span>
      <span class="rejected">&#10007; ${stats.rejected} rejected</span>
      <span>${stats.remaining} remaining</span>
    </div>

    <div class="top10">
      <div class="label">Top 10 by confidence</div>
      <div class="top10-list">
        ${top10Html()}
      </div>
    </div>

    <div class="waveform-card">
      <div class="label">Full waveform</div>
      <div id="waveform"></div>
    </div>

    <div class="nav">
      <button class="nav-btn" onclick="go(-1)" ${idx === 0 ? 'disabled' : ''}>&larr; Prev</button>
      <span class="nav-counter">${idx + 1} / ${D.total}</span>
      <button class="nav-btn" onclick="go(1)" ${idx === D.total - 1 ? 'disabled' : ''}>Next &rarr;</button>
      <button class="nav-btn" onclick="goNextUndecided()">Next undecided</button>
    </div>

    <div class="pred-info">
      <div class="pred-title">
        <span class="technique">${pred.label.replace(/_/g, ' ')}</span>
        <span class="conf">${(pred.confidence * 100).toFixed(0)}% confidence</span>
      </div>
      <div class="pred-meta">
        ${pred.start.toFixed(2)}s &ndash; ${pred.end.toFixed(2)}s
        &nbsp;(${(pred.end - pred.start).toFixed(2)}s)
      </div>
      ${dec ? `<div class="pred-status ${dec.action}">${statusText(dec)}</div>` : ''}
    </div>

    <div class="audio-section">
      <div class="audio-row">
        <span class="audio-label">Segment</span>
        <audio id="seg-audio" controls preload="auto" style="height:32px; flex:1;"></audio>
      </div>
      <div class="audio-row">
        <span class="audio-label">Reference</span>
        <audio id="ref-audio" controls preload="auto" style="height:32px; flex:1;"></audio>
      </div>
    </div>

    <div class="relabel-section">
      <label>Listen to / relabel as a different technique</label>
      <div class="relabel-row">
        <select class="relabel-select" id="relabel-select" onchange="onRelabelSelect()">
          ${D.labels.map(l =>
            `<option value="${l}" ${l === pred.label ? 'selected' : ''}>${l.replace(/_/g, ' ')}</option>`
          ).join('')}
        </select>
        <button class="listen-btn" onclick="listenRef()">Listen</button>
      </div>
    </div>

    <div class="actions">
      <button class="btn btn-confirm" onclick="doConfirm()">&#10003; Confirm</button>
      <button class="btn btn-confirm" onclick="doRelabel()" style="background:#1e3a5f;">&#9998; Relabel</button>
      <button class="btn btn-reject" onclick="doReject()">&#10007; Reject</button>
      ${dec ? '<button class="btn btn-undo" onclick="doUndo()">Undo</button>' : ''}
    </div>

    <details style="margin-bottom:16px;">
      <summary style="color:var(--dim); cursor:pointer; font-size:13px; margin-bottom:8px;">
        All predictions (click to jump)
      </summary>
      <div class="pred-list">
        ${D.predictions.map((p, i) => {
          const d = decisions[String(i)];
          let dotClass = 'dot-pending';
          if (d) {
            if (d.action === 'confirm') dotClass = 'dot-confirmed';
            else if (d.action === 'relabel') dotClass = 'dot-relabeled';
            else if (d.action === 'reject') dotClass = 'dot-rejected';
          }
          return `<div class="pred-list-item ${i === idx ? 'active' : ''}" onclick="jump(${i})">
            <span class="dot ${dotClass}"></span>
            <span class="lbl">${p.label.replace(/_/g, ' ')}</span>
            <span class="time">${p.start.toFixed(2)}s</span>
          </div>`;
        }).join('')}
      </div>
    </details>

    <div class="finish-section">
      <button class="btn btn-save" onclick="doSave()">Save corrections</button>
      ${D.canRetrain ? '<button class="btn btn-retrain" onclick="doRetrain()">Retrain model</button>' : ''}
      <div class="finish-msg" id="finish-msg"></div>
    </div>
  `;

  loadWaveform();
  loadSegmentAudio();
  loadRefAudio(pred.label);
}

async function loadSegmentAudio() {
  const audio = document.getElementById('seg-audio');
  if (!audio) return;
  const url = await getSegment(idx);
  audio.src = url;
}

function loadRefAudio(label) {
  const audio = document.getElementById('ref-audio');
  if (!audio) return;
  const url = D.refUrls[label];
  if (url) {
    audio.src = url;
  } else {
    audio.removeAttribute('src');
  }
}

function statusText(dec) {
  if (dec.action === 'confirm') return '&#10003; Confirmed';
  if (dec.action === 'relabel') return `&#9998; Relabeled → ${dec.label.replace(/_/g, ' ')}`;
  if (dec.action === 'reject') return '&#10007; Rejected';
  return '';
}

function computeStats() {
  let confirmed = 0, relabeled = 0, rejected = 0;
  for (const d of Object.values(decisions)) {
    if (d.action === 'confirm') confirmed++;
    else if (d.action === 'relabel') relabeled++;
    else if (d.action === 'reject') rejected++;
  }
  return { confirmed, relabeled, rejected, remaining: D.total - confirmed - relabeled - rejected };
}

// ---- Actions ----

function go(delta) {
  const next = idx + delta;
  if (next >= 0 && next < D.total) { idx = next; render(); }
}

function jump(i) { idx = i; render(); }

function goNextUndecided() {
  for (let i = idx + 1; i < D.total; i++) {
    if (!decisions[String(i)]) { idx = i; render(); return; }
  }
  for (let i = 0; i < idx; i++) {
    if (!decisions[String(i)]) { idx = i; render(); return; }
  }
}

async function doConfirm() {
  await fetch(`/api/confirm/${idx}`, { method: 'POST' });
  await refreshDecisions();
  render();
}

async function doRelabel() {
  const sel = document.getElementById('relabel-select');
  const label = sel ? sel.value : null;
  await fetch(`/api/confirm/${idx}?label=${encodeURIComponent(label)}`, { method: 'POST' });
  await refreshDecisions();
  render();
}

async function doReject() {
  await fetch(`/api/reject/${idx}`, { method: 'POST' });
  await refreshDecisions();
  render();
}

async function doUndo() {
  await fetch(`/api/clear/${idx}`, { method: 'POST' });
  await refreshDecisions();
  render();
}

function onRelabelSelect() {
  const sel = document.getElementById('relabel-select');
  if (sel) loadRefAudio(sel.value);
}

function listenRef() {
  const audio = document.getElementById('ref-audio');
  if (audio && audio.src) {
    audio.currentTime = 0;
    audio.play();
  }
}

async function doSave() {
  const msg = document.getElementById('finish-msg');
  if (msg) msg.textContent = 'Saving...';
  const r = await (await fetch('/api/save', { method: 'POST' })).json();
  if (msg) msg.textContent = `Saved ${r.count} labels → ${r.path}`;
}

async function doRetrain() {
  const msg = document.getElementById('finish-msg');
  if (msg) msg.textContent = 'Retraining (this may take a while)...';
  const r = await (await fetch('/api/retrain', { method: 'POST' })).json();
  if (msg) msg.textContent = r.message;
}

// ---- Keyboard shortcuts ----
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') go(-1);
  else if (e.key === 'ArrowRight') go(1);
  else if (e.key === 'c' || e.key === 'y') doConfirm();
  else if (e.key === 'x' || e.key === 'n') doReject();
  else if (e.key === 'u') doUndo();
  else if (e.key === 's') { const a = document.getElementById('seg-audio'); if (a) { a.currentTime = 0; a.play(); } }
  else if (e.key === 'r') { const a = document.getElementById('ref-audio'); if (a) { a.currentTime = 0; a.play(); } }
  else if (e.key === 'Tab') { e.preventDefault(); goNextUndecided(); }
});

init();
</script>
</body>
</html>
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prediction reviewer")
    parser.add_argument("--audio", required=True, help="Audio file")
    parser.add_argument("--model", default=None, help="Model .pt file")
    parser.add_argument("--predictions", default=None,
                        help="Pre-computed predictions JSON")
    parser.add_argument("--ref_dir", default=None,
                        help="Directory of reference technique WAVs")
    parser.add_argument("--train_dir", default=None,
                        help="Directory to accumulate corrected data")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: <stem>_labels.json)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
        predictions.sort(key=lambda p: (p["start"], p["end"]))
    elif args.model:
        print("Running inference...")
        predictions = run_inference(args.audio, args.model, threshold=args.threshold)
    else:
        parser.error("Provide --model or --predictions")

    print(f"{len(predictions)} predictions")
    top10 = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)[:10]
    for p in top10:
        print(f"  {p['label']:20s}  {p['start']:.2f}s-{p['end']:.2f}s  ({p.get('confidence',0):.0%})")
    if len(predictions) > 10:
        print(f"  ... and {len(predictions) - 10} more")

    out_path = args.out or str(Path(args.audio).with_suffix("")) + "_labels.json"

    state = ReviewState(
        audio_path=args.audio,
        predictions=predictions,
        out_path=out_path,
        ref_dir=args.ref_dir,
        train_dir=args.train_dir,
        model_path=args.model,
    )

    segments, ref_urls = precompute(state.waveform, state.sr, state.predictions, args.ref_dir)

    app = create_app(state, segments, ref_urls)
    print(f"\nOpen http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
