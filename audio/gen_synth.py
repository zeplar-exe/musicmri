"""
Generate synthetic technique pieces.

Produces:
    data_dir/audio/NNNN.wav
    data_dir/labels/NNNN.json

Each example is a longer piece (PIECE_DUR_RANGE seconds) built by laying several
technique EVENTS along a timeline: events are separated by pauses, may overlap to
create simultaneity, and never have more than MAX_CONCURRENT labels sounding at
once. The matching JSON lists every technique's time span.

Usage:
    python gen_synth.py --out_dir ./data --n 1000

> Thanks Claude
"""

import argparse
import json
import os
import glob
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt

SR = 44100
RNG = np.random.default_rng(42)

# Background music pool (loaded lazily)
_bg_pool = []  # list of (samples,) float64 arrays at SR


def load_backgrounds(bg_dir):
    """Load all audio files from a directory into the background pool."""
    global _bg_pool
    _bg_pool = []
    if not bg_dir or not os.path.isdir(bg_dir):
        return
    import librosa
    patterns = ["*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(bg_dir, pat)))
    files.sort()
    for path in files:
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
            if len(y) > SR * 2:  # at least 2 seconds
                _bg_pool.append(y)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")
    print(f"  Loaded {len(_bg_pool)} background tracks")


def mix_background(sig, snr_db):
    """Mix a random chunk from the background pool under the signal."""
    if not _bg_pool:
        return sig
    bg = _bg_pool[RNG.integers(len(_bg_pool))]
    # Pick a random offset that fits our signal length
    max_start = len(bg) - len(sig)
    if max_start <= 0:
        # Background shorter than signal — loop it
        reps = (len(sig) // len(bg)) + 1
        bg = np.tile(bg, reps)
        max_start = len(bg) - len(sig)
    start = RNG.integers(0, max(max_start, 1))
    chunk = bg[start:start + len(sig)]
    if len(chunk) < len(sig):
        chunk = np.pad(chunk, (0, len(sig) - len(chunk)))
    # Mix at specified SNR
    sig_power = np.mean(sig ** 2) + 1e-10
    bg_power = np.mean(chunk ** 2) + 1e-10
    scale = np.sqrt(sig_power / (bg_power * 10 ** (snr_db / 10)))
    return sig + chunk * scale


# ============================================================
# Primitives
# ============================================================

def t(duration):
    return np.linspace(0, duration, int(SR * duration), endpoint=False)


def sine(freq, duration):
    return np.sin(2 * np.pi * freq * t(duration))


def sawtooth(freq, duration, n_harmonics=12):
    tt = t(duration)
    out = np.zeros(len(tt))
    for k in range(1, n_harmonics + 1):
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * tt) / k
    return out * (2 / np.pi)


def pick_waveform(freq, duration):
    """Randomly choose sine or sawtooth."""
    if RNG.random() < 0.5:
        return sine(freq, duration)
    return sawtooth(freq, duration)


def lowpass(signal, cutoff, order=4):
    sos = butter(order, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, signal)


def bandpass(signal, low, high, order=2):
    sos = butter(order, [low, high], btype='band', fs=SR, output='sos')
    return sosfilt(sos, signal)


def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return signal
    return signal / peak


def add_noise(signal, snr_db=30):
    """Add white noise at given SNR."""
    noise = RNG.normal(0, 1, len(signal))
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + noise * np.sqrt(noise_power)


# ============================================================
# Technique generators
#
# Each returns (signal, annotations) where annotations is a list of
# {"label": str, "start": float, "end": float}
#
# All operate on a single base frequency and duration.
# ============================================================

# --- Dynamics ---

def gen_crescendo(freq, dur):
    tt = t(dur)
    env = np.linspace(0.05, 1.0, len(tt))
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "crescendo", "start": 0.0, "end": dur}]


def gen_decrescendo(freq, dur):
    tt = t(dur)
    env = np.linspace(1.0, 0.05, len(tt))
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "decrescendo", "start": 0.0, "end": dur}]


def gen_swell(freq, dur):
    tt = t(dur)
    mid = len(tt) // 2
    env = np.concatenate([
        np.linspace(0.05, 1.0, mid),
        np.linspace(1.0, 0.05, len(tt) - mid)
    ])
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "swell", "start": 0.0, "end": dur}]


def gen_subito_piano(freq, dur):
    tt = t(dur)
    env = np.ones(len(tt))
    mid = len(tt) // 2
    env[mid:] = 0.15
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "subito_piano", "start": mid / SR, "end": dur}]


# --- Ornaments ---

def gen_vibrato(freq, dur):
    tt = t(dur)
    vib_rate = RNG.uniform(4, 7)
    # Depth as a fraction of pitch (~constant cents) so vibrato looks the same
    # whether the note is 60 Hz or 1200 Hz, instead of absolute Hz.
    vib_depth = RNG.uniform(0.012, 0.04)  # ~20-70 cents
    inst_freq = freq * (1 + vib_depth * np.sin(2 * np.pi * vib_rate * tt))
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "vibrato", "start": 0.0, "end": dur}]


def gen_trill(freq, dur):
    tt = t(dur)
    interval = RNG.choice([1, 2])
    freq_upper = freq * (2 ** (interval / 12))
    rate = RNG.uniform(6, 12)
    switch = (np.sin(2 * np.pi * rate * tt) > 0).astype(float)
    inst_freq = freq * (1 - switch) + freq_upper * switch
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "trill", "start": 0.0, "end": dur}]


def gen_glissando(freq, dur):
    tt = t(dur)
    direction = RNG.choice([-1, 1])
    interval = RNG.uniform(5, 12)
    freq_end = freq * (2 ** (direction * interval / 12))
    freqs = freq * (freq_end / freq) ** (tt / dur)
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    sig = np.sin(phase)
    return sig, [{"label": "glissando", "start": 0.0, "end": dur}]


def gen_grace_note(freq, dur):
    tt = t(dur)
    grace_freq = freq / (2 ** (RNG.choice([1, 2]) / 12))
    grace_dur = RNG.uniform(0.04, 0.08)
    grace_samples = int(SR * grace_dur)
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[:grace_samples] = grace_freq
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "grace_note", "start": 0.0, "end": grace_dur}]


def gen_pitch_bend(freq, dur):
    tt = t(dur)
    bend_semi = RNG.uniform(1, 3)
    bend_freq = freq * (2 ** (bend_semi / 12))
    n = len(tt)

    if RNG.random() < 0.5:
        # Bend up and release: a continuous triangular arc, no steady tone,
        # so labeling the whole span is honest.
        r_up = int(n * RNG.uniform(0.45, 0.55))
        freq_env = np.concatenate([
            np.linspace(freq, bend_freq, r_up),
            np.linspace(bend_freq, freq, n - r_up),
        ])
        label_end = dur
    else:
        # Bend up and sustain: only the ramp is pitch motion; the held bent
        # note that follows is a steady tone and stays unlabeled.
        r_up = int(n * RNG.uniform(0.25, 0.45))
        freq_env = np.concatenate([
            np.linspace(freq, bend_freq, r_up),
            np.ones(n - r_up) * bend_freq,
        ])
        label_end = r_up / SR

    phase = 2 * np.pi * np.cumsum(freq_env) / SR
    sig = np.sin(phase)
    return sig, [{"label": "pitch_bend", "start": 0.0, "end": round(label_end, 4)}]


# --- Articulation (phrase-based) ---

def _gen_phrase(freq, dur, notes, duty, label):
    slot = dur / notes
    out = np.zeros(int(SR * dur))
    for i in range(notes):
        start = int(SR * slot * i)
        note_dur = slot * duty
        seg = pick_waveform(freq, note_dur)
        fade = int(SR * 0.01)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out, [{"label": label, "start": 0.0, "end": dur}]


def gen_staccato(freq, dur):
    notes = RNG.integers(6, 12)
    return _gen_phrase(freq, dur, notes, duty=0.3, label="staccato")


def gen_legato(freq, dur):
    notes = RNG.integers(6, 12)
    return _gen_phrase(freq, dur, notes, duty=0.95, label="legato")


def gen_accent(freq, dur):
    notes = 8
    slot = dur / notes
    out = np.zeros(int(SR * dur))
    accent_idx = RNG.integers(1, notes - 1)
    for i in range(notes):
        start = int(SR * slot * i)
        seg = pick_waveform(freq, slot * 0.7)
        seg *= (1.5 if i == accent_idx else 0.5)
        fade = int(SR * 0.01)
        if fade > 0 and fade < len(seg):
            seg[-fade:] *= np.linspace(1, 0, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    accent_start = accent_idx * slot
    return out, [{"label": "accent", "start": round(accent_start, 3),
                  "end": round(accent_start + slot, 3)}]


# --- Tempo modification (phrase-based) ---

def _gen_tempo_phrase(freq, dur, notes, ioi_pattern, label):
    iois = ioi_pattern / ioi_pattern.sum() * dur
    out = np.zeros(int(SR * dur))
    pos = 0.0
    for ioi in iois:
        note_dur = ioi * 0.7
        start = int(SR * pos)
        seg = pick_waveform(freq, note_dur)
        fade = min(int(SR * 0.01), len(seg) // 4)
        if fade > 0:
            seg[-fade:] *= np.linspace(1, 0, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out, [{"label": label, "start": 0.0, "end": dur}]


def gen_accelerando(freq, dur):
    notes = RNG.integers(8, 14)
    iois = np.linspace(1.0, 0.3, notes)
    return _gen_tempo_phrase(freq, dur, notes, iois, "accelerando")


def gen_ritardando(freq, dur):
    notes = RNG.integers(8, 14)
    iois = np.linspace(0.3, 1.0, notes)
    return _gen_tempo_phrase(freq, dur, notes, iois, "ritardando")


def gen_fermata(freq, dur):
    notes = 6
    iois = np.ones(notes)
    ferm_idx = RNG.integers(1, notes - 1)
    iois[ferm_idx] = 4.0
    iois = iois / iois.sum() * dur
    out = np.zeros(int(SR * dur))
    pos = 0.0
    ferm_start = 0.0
    ferm_end = 0.0
    for i, ioi in enumerate(iois):
        if i == ferm_idx:
            ferm_start = pos
            ferm_end = pos + ioi
        start = int(SR * pos)
        seg = pick_waveform(freq, ioi * 0.85)
        fade = min(int(SR * 0.02), len(seg) // 4)
        if fade > 0:
            seg[-fade:] *= np.linspace(1, 0, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out, [{"label": "fermata", "start": round(ferm_start, 3),
                  "end": round(ferm_end, 3)}]


def gen_caesura(freq, dur):
    notes = 8
    slot = dur / notes
    out = np.zeros(int(SR * dur))
    pause_at = RNG.integers(2, notes - 2)
    for i in range(notes):
        if i == pause_at:
            continue
        start = int(SR * slot * i)
        seg = pick_waveform(freq, slot * 0.7)
        fade = min(int(SR * 0.01), len(seg) // 4)
        if fade > 0:
            seg[-fade:] *= np.linspace(1, 0, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    pause_start = pause_at * slot
    return out, [{"label": "caesura", "start": round(pause_start, 3),
                  "end": round(pause_start + slot, 3)}]


def gen_scoop(freq, dur):
    """Approach a note from below."""
    tt = t(dur)
    scoop_dur = RNG.uniform(0.05, 0.15)
    scoop_samples = int(SR * scoop_dur)
    semitones_below = RNG.uniform(2, 5)
    start_freq = freq / (2 ** (semitones_below / 12))
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[:scoop_samples] = np.linspace(start_freq, freq, scoop_samples)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "scoop", "start": 0.0, "end": scoop_dur}]


def gen_fall_off(freq, dur):
    """Pitch drops off at the end of a note."""
    tt = t(dur)
    fall_dur = RNG.uniform(0.08, 0.2)
    fall_samples = int(SR * fall_dur)
    semitones_drop = RNG.uniform(3, 8)
    end_freq = freq / (2 ** (semitones_drop / 12))
    inst_freq = np.ones(len(tt)) * freq
    fall_start = max(0, len(tt) - fall_samples)
    inst_freq[fall_start:] = np.linspace(freq, end_freq, len(tt) - fall_start)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    # Fade out during fall
    env = np.ones(len(tt))
    env[fall_start:] = np.linspace(1.0, 0.1, len(tt) - fall_start)
    sig *= env
    fall_start_sec = fall_start / SR
    return sig, [{"label": "fall_off", "start": round(fall_start_sec, 4), "end": dur}]


# --- Percussion / extended ---

def gen_roll(freq, dur):
    """Rapid repeated strikes (drum roll)."""
    # Lower-pitched drums roll slower; high-pitched can roll faster
    rate = RNG.uniform(8, 30) if freq > 200 else RNG.uniform(5, 15)
    n_hits = int(rate * dur)
    out = np.zeros(int(SR * dur))
    # Scale hit duration with frequency: need at least 3-4 cycles to register
    min_hit_dur = max(4.0 / freq, 0.015)
    hit_dur = RNG.uniform(min_hit_dur, min_hit_dur * 2.5)
    # Low freqs have more boom (tone), high freqs have more snap (noise)
    noise_mix = np.clip(0.1 + (freq - 80) / 1000, 0.05, 0.5)
    tone_mix = 1.0 - noise_mix
    # Decay rate: slower for low freqs so the tone develops
    decay_rate = np.clip(10 + (freq - 80) / 30, 8, 40)
    for i in range(n_hits):
        pos = i / rate
        start = int(SR * pos)
        n_samp = int(SR * hit_dur)
        if start + n_samp > len(out):
            break
        tt_hit = np.linspace(0, hit_dur, n_samp)
        hit = RNG.normal(0, 1, n_samp) * noise_mix + np.sin(2 * np.pi * freq * tt_hit) * tone_mix
        hit *= np.exp(-decay_rate * np.linspace(0, 1, n_samp))
        hit *= RNG.uniform(0.6, 1.0)
        out[start:start + n_samp] += hit
    return out, [{"label": "roll", "start": 0.0, "end": dur}]


def gen_choke(freq, dur):
    """Abrupt muting of a resonant sound."""
    tt = t(dur)
    # Ring up then cut
    ring_dur = RNG.uniform(0.3, 0.6) * dur
    ring_samples = int(SR * ring_dur)
    sig = pick_waveform(freq, dur)
    # Add some shimmer (detuned harmonics)
    sig += 0.3 * np.sin(2 * np.pi * freq * 2.01 * tt)
    # Abrupt cut
    cut_dur = RNG.uniform(0.01, 0.03)
    cut_samples = int(SR * cut_dur)
    env = np.ones(len(tt))
    cut_start = ring_samples
    cut_end = min(cut_start + cut_samples, len(tt))
    env[cut_start:cut_end] = np.linspace(1.0, 0.0, cut_end - cut_start)
    env[cut_end:] = 0.0
    sig *= env
    choke_start = ring_samples / SR
    return sig, [{"label": "choke", "start": round(choke_start, 4), "end": round(choke_start + cut_dur, 4)}]


def gen_arpeggio(freq, dur):
    """Broken chord: notes of a chord played in sequence."""
    intervals = [0, 4, 7, 12]  # major chord
    if RNG.random() < 0.3:
        intervals = [0, 3, 7, 12]  # minor chord
    n_notes = len(intervals)
    note_dur = dur / n_notes
    out = np.zeros(int(SR * dur))
    for i, semi in enumerate(intervals):
        note_freq = freq * (2 ** (semi / 12))
        start = int(SR * note_dur * i)
        seg = pick_waveform(note_freq, note_dur)
        # Pluck-like decay
        decay = np.exp(-RNG.uniform(2, 5) * np.linspace(0, 1, len(seg)))
        seg *= decay
        fade = int(SR * 0.005)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out, [{"label": "arpeggio", "start": 0.0, "end": dur}]


# ============================================================
# Generator registry
# ============================================================

# Category -> list of generators
# Used for composing compatible techniques
GENERATORS = {
    "dynamics": [
        gen_crescendo, gen_decrescendo, gen_swell, gen_subito_piano, gen_accent,
    ],
    "ornaments": [
        gen_vibrato, gen_trill, gen_glissando,
        gen_grace_note, gen_pitch_bend,
        gen_scoop, gen_fall_off, gen_arpeggio,
    ],
    "articulation": [
        gen_staccato, gen_legato,
    ],
    "tempo": [
        gen_accelerando, gen_ritardando, gen_fermata, gen_caesura,
    ],
    "percussion": [
        gen_roll, gen_choke,
    ],
}

ALL_GENERATORS = []
for gens in GENERATORS.values():
    ALL_GENERATORS.extend(gens)


# ============================================================
# Composable overlays (applied post-hoc to a base signal)
# ============================================================

def overlay_crescendo(sig, dur):
    env = np.linspace(0.05, 1.0, len(sig))
    return sig * env, {"label": "crescendo", "start": 0.0, "end": dur}

def overlay_decrescendo(sig, dur):
    env = np.linspace(1.0, 0.05, len(sig))
    return sig * env, {"label": "decrescendo", "start": 0.0, "end": dur}

def overlay_swell(sig, dur):
    n = len(sig)
    mid = n // 2
    env = np.concatenate([np.linspace(0.05, 1.0, mid), np.linspace(1.0, 0.05, n - mid)])
    return sig * env, {"label": "swell", "start": 0.0, "end": dur}

def overlay_vibrato(sig, dur, freq):
    """Impart vibrato on an existing signal via a modulated fractional delay,
    preserving the base technique's amplitude envelope and articulation.

    (freq is unused — kept for the generic freq-overlay call signature.)
    """
    n = len(sig)
    tt = t(dur)[:n]
    vib_rate = RNG.uniform(4, 7)
    depth_samp = RNG.uniform(0.3, 1.2) / 1000 * SR   # 0.3-1.2 ms delay swing
    mod = depth_samp * (1 + np.sin(2 * np.pi * vib_rate * tt)) / 2  # 0..depth
    idx = np.arange(n) - mod
    np.clip(idx, 0, n - 1, out=idx)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = idx - i0
    out = sig[i0] * (1 - frac) + sig[i1] * frac
    return out, {"label": "vibrato", "start": 0.0, "end": dur}

ALL_OVERLAYS = [
    ("amp",  overlay_crescendo),
    ("amp",  overlay_decrescendo),
    ("amp",  overlay_swell),
    ("freq", overlay_vibrato),
]

# Which overlay kinds are compatible with each base category.
COMPATIBLE_KINDS = {
    "ornaments":      {"amp"},
    "articulation":   {"amp", "freq"},
    "tempo":          {"amp", "freq"},
    "dynamics":       {"freq"},
    "percussion":     {"amp"},
}


# ============================================================
# Sample generation
# ============================================================

# Round-robin index for uniform base generator sampling
_gen_cycle_idx = 0

def generate_single(freq, dur):
    """Generate a single-technique example, cycling through generators uniformly."""
    global _gen_cycle_idx
    gen_fn = ALL_GENERATORS[_gen_cycle_idx % len(ALL_GENERATORS)]
    _gen_cycle_idx += 1
    sig, anns = gen_fn(freq, dur)
    return sig, anns


def generate_composed(freq, dur):
    """Generate a multi-technique example by combining a base generator
    with 1-2 compatible overlays picked uniformly from the flat pool."""
    base_cat = RNG.choice(list(COMPATIBLE_KINDS.keys()))
    gen_fn = RNG.choice(GENERATORS[base_cat])
    sig, anns = gen_fn(freq, dur)

    existing_labels = {a["label"] for a in anns}
    allowed_kinds = COMPATIBLE_KINDS[base_cat]
    candidates = [(k, fn) for k, fn in ALL_OVERLAYS if k in allowed_kinds]
    RNG.shuffle(candidates)

    n_overlays = RNG.choice([1, 2], p=[0.6, 0.4])
    applied = 0
    for kind, fn in candidates:
        if applied >= n_overlays:
            break

        if kind == "freq":
            new_sig, ann = fn(sig, dur, freq)
        else:
            new_sig, ann = fn(sig, dur)

        if ann["label"] in existing_labels:
            continue

        sig = new_sig
        anns.append(ann)
        existing_labels.add(ann["label"])
        applied += 1

    return sig, anns


_label_counts = {}

# Piece-level layout. A "piece" is a longer timeline onto which several technique
# EVENTS are placed, separated by pauses and allowed to overlap up to a cap.
PIECE_DUR_RANGE = (8.0, 20.0)     # seconds of synthesized audio per file
EVENT_DUR_RANGE = (1.0, 3.0)      # length of one technique event
PAUSE_RANGE = (0.3, 1.5)          # silence inserted between events
PAUSE_PROB = 0.5                  # chance of a pause before the next event
OVERLAP_PROB = 0.4                # chance an event overlaps the previous one
MAX_CONCURRENT = 3                # never more than this many labels sounding at once
MAX_EVENTS = 40                   # safety cap on events per piece
# A piece emits ~EXPECTED_EVENTS_PER_PIECE times more labels than the old
# one-technique-per-file scheme, so scale the imbalance cap to match.
EXPECTED_EVENTS_PER_PIECE = 6


def _balance_anns(anns, n_total):
    """Drop overrepresented overlay labels (keep the base technique, anns[0]) and
    bump the running counts. Same imbalance guard as before, scaled per piece."""
    if len(anns) <= 1:
        kept = anns
    else:
        cap = (n_total * EXPECTED_EVENTS_PER_PIECE / len(ALL_GENERATORS)) * 1.2
        kept = [anns[0]]
        for ann in anns[1:]:
            if _label_counts.get(ann["label"], 0) < cap:
                kept.append(ann)
    for ann in kept:
        _label_counts[ann["label"]] = _label_counts.get(ann["label"], 0) + 1
    return kept


def _peak_concurrent(events, start, end):
    """Max labels already sounding at any instant within [start, end). Concurrency
    only steps up at an event's start, so checking those points (plus the window
    start) catches the peak."""
    points = [start] + [e["start"] for e in events if start <= e["start"] < end]
    peak = 0
    for p in points:
        active = sum(e["nlabels"] for e in events if e["start"] <= p < e["end"])
        peak = max(peak, active)
    return peak


def _make_event(n_total):
    """One technique event: pick freq + duration, generate, balance labels, and
    peak-normalize. Per-event normalization keeps a quiet technique from vanishing
    once several events are summed into the piece. Returns (signal, anns, dur)."""
    freq = RNG.uniform(60, 1200)
    dur = RNG.uniform(*EVENT_DUR_RANGE)
    if RNG.random() < 0.5:
        sig, anns = generate_single(freq, dur)
    else:
        sig, anns = generate_composed(freq, dur)
    anns = _balance_anns(anns, n_total)
    sig = normalize(sig) * RNG.uniform(0.6, 1.0)
    return sig, anns, dur


def _schedule(total_dur, n_total):
    """Lay technique events along a timeline: walk a cursor left to right, drop in
    pauses, and sometimes start an event early so it overlaps the previous one --
    but only while total simultaneous labels stay <= MAX_CONCURRENT."""
    events = []
    cursor = 0.0
    while cursor < total_dur - EVENT_DUR_RANGE[0] and len(events) < MAX_EVENTS:
        if events and RNG.random() < PAUSE_PROB:
            cursor += RNG.uniform(*PAUSE_RANGE)
        if cursor >= total_dur - EVENT_DUR_RANGE[0]:
            break

        sig, anns, dur = _make_event(n_total)
        dur = min(dur, total_dur - cursor)        # keep the event inside the piece
        sig = sig[:int(SR * dur)]
        k = len(anns)

        start = cursor
        if events and RNG.random() < OVERLAP_PROB:  # try to overlap the previous event
            prev = events[-1]
            cand = RNG.uniform(prev["start"], prev["end"])
            if _peak_concurrent(events, cand, cand + dur) + k <= MAX_CONCURRENT:
                start = cand

        for a in anns:
            a["end"] = min(a["end"], dur)          # the signal was clamped; clamp labels too
            a["start"] += start
            a["end"] += start
        events.append({"start": start, "end": start + dur,
                       "sig": sig, "anns": anns, "nlabels": k})
        cursor = max(cursor, start + dur)
    return events


def generate_sample(idx, n_total):
    """Generate one longer piece: several technique events placed along a timeline
    with pauses between them and at most MAX_CONCURRENT labels sounding at once."""
    total_dur = RNG.uniform(*PIECE_DUR_RANGE)
    events = _schedule(total_dur, n_total)

    piece_len = max(e["end"] for e in events)
    out = np.zeros(int(SR * piece_len))
    anns = []
    for e in events:
        s = int(SR * e["start"])
        seg = e["sig"]
        end = min(s + len(seg), len(out))
        out[s:end] += seg[:end - s]
        anns.extend(e["anns"])

    # Mix background music (if available) at random SNR
    if _bg_pool:
        bg_snr = RNG.uniform(-5, 15)  # technique can be quieter than background
        out = mix_background(out, bg_snr)

    out = add_noise(out, snr_db=RNG.uniform(20, 50))
    out = normalize(out)

    for ann in anns:
        ann["start"] = round(ann["start"], 4)
        ann["end"] = round(min(ann["end"], piece_len), 4)

    return out, anns, piece_len


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./data",
                        help="Output directory for audio/ and labels/")
    parser.add_argument("-n", type=int, default=3000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background_dir", default=None,
                        help="Directory of MP3s/WAVs to mix under synthetic samples")
    args = parser.parse_args()

    global RNG, _label_counts, _gen_cycle_idx
    RNG = np.random.default_rng(args.seed)
    _label_counts = {}
    _gen_cycle_idx = 0

    if args.background_dir:
        print(f"Loading background audio from {args.background_dir}...")
        load_backgrounds(args.background_dir)

    audio_dir = os.path.join(args.out_dir, "audio")
    label_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    label_counts = {}

    for i in range(args.n):
        sig, anns, dur = generate_sample(i, args.n)

        # Save audio
        pcm = (sig * 32767 * 0.8).astype(np.int16)
        wav_path = os.path.join(audio_dir, f"{i:04d}.wav")
        write(wav_path, SR, pcm)

        # Save labels
        json_path = os.path.join(label_dir, f"{i:04d}.json")
        with open(json_path, "w") as f:
            json.dump(anns, f, indent=2)

        # Track counts
        for ann in anns:
            label_counts[ann["label"]] = label_counts.get(ann["label"], 0) + 1

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.n}")

    print(f"\nDone. {args.n} samples in {args.out_dir}/")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count}")


if __name__ == "__main__":
    main()