"""
Generate synthetic training data for the technique CNN.

Produces:
    data_dir/audio/NNNN.wav
    data_dir/labels/NNNN.json

Each example is 2-4 seconds of synthesized audio with 1-3 techniques applied,
and a matching JSON label file with time spans.

Usage:
    python generate_data.py --out_dir ./data --n 1000
"""

import argparse
import json
import os
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt

SR = 44100
RNG = np.random.default_rng(42)


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


def gen_sforzando(freq, dur):
    tt = t(dur)
    env = np.ones(len(tt)) * 0.15
    attack_len = int(len(tt) * RNG.uniform(0.05, 0.15))
    env[:attack_len] = np.exp(-5 * np.linspace(0, 1, attack_len))
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "sforzando", "start": 0.0, "end": attack_len / SR}]


def gen_swell(freq, dur):
    tt = t(dur)
    mid = len(tt) // 2
    env = np.concatenate([
        np.linspace(0.05, 1.0, mid),
        np.linspace(1.0, 0.05, len(tt) - mid)
    ])
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "swell", "start": 0.0, "end": dur}]


def gen_fortepiano(freq, dur):
    tt = t(dur)
    env = np.ones(len(tt)) * 0.15
    attack_end = int(len(tt) * 0.05)
    env[:attack_end] = 1.0
    drop = int(len(tt) * 0.08)
    env[attack_end:attack_end + drop] = np.linspace(1.0, 0.15, drop)
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "fortepiano", "start": 0.0, "end": dur}]


def gen_subito_forte(freq, dur):
    tt = t(dur)
    env = np.ones(len(tt)) * 0.15
    mid = len(tt) // 2
    env[mid:] = 1.0
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "subito_forte", "start": mid / SR, "end": dur}]


def gen_subito_piano(freq, dur):
    tt = t(dur)
    env = np.ones(len(tt))
    mid = len(tt) // 2
    env[mid:] = 0.15
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "subito_piano", "start": mid / SR, "end": dur}]


def gen_morendo(freq, dur):
    tt = t(dur)
    env = np.exp(-RNG.uniform(2, 5) * tt / dur)
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "morendo", "start": 0.0, "end": dur}]


# --- Ornaments ---

def gen_vibrato(freq, dur):
    tt = t(dur)
    vib_rate = RNG.uniform(4, 7)
    vib_depth = RNG.uniform(3, 15)
    inst_freq = freq + vib_depth * np.sin(2 * np.pi * vib_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "vibrato", "start": 0.0, "end": dur}]


def gen_tremolo(freq, dur):
    tt = t(dur)
    trem_rate = RNG.uniform(5, 10)
    trem_depth = RNG.uniform(0.5, 0.9)
    env = 1.0 - trem_depth * 0.5 * (1 + np.sin(2 * np.pi * trem_rate * tt))
    sig = env * pick_waveform(freq, dur)
    return sig, [{"label": "tremolo", "start": 0.0, "end": dur}]


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


def gen_mordent(freq, dur):
    tt = t(dur)
    lower = freq / (2 ** (RNG.choice([1, 2]) / 12))
    mordent_dur = RNG.uniform(0.06, 0.12)
    mordent_samples = int(SR * mordent_dur)
    seg = mordent_samples // 3
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[seg:2 * seg] = lower
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "mordent", "start": 0.0, "end": mordent_dur}]


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
    r_up = int(n * 0.3)
    hold = int(n * 0.4)
    r_dn = n - r_up - hold
    freq_env = np.concatenate([
        np.linspace(freq, bend_freq, r_up),
        np.ones(hold) * bend_freq,
        np.linspace(bend_freq, freq, r_dn)
    ])
    phase = 2 * np.pi * np.cumsum(freq_env) / SR
    sig = np.sin(phase)
    return sig, [{"label": "pitch_bend", "start": 0.0, "end": dur}]


def gen_turn(freq, dur):
    tt = t(dur)
    above = freq * (2 ** (2 / 12))
    below = freq / (2 ** (2 / 12))
    orn_dur = RNG.uniform(0.15, 0.25)
    orn = int(SR * orn_dur)
    seg = orn // 4
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[0:seg] = above
    inst_freq[seg:2 * seg] = freq
    inst_freq[2 * seg:3 * seg] = below
    inst_freq[3 * seg:4 * seg] = freq
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "turn", "start": 0.0, "end": orn_dur}]


def gen_portamento(freq, dur):
    tt = t(dur)
    target = freq * (2 ** (RNG.uniform(3, 7) / 12))
    slide_dur = RNG.uniform(0.15, 0.35)
    slide_samples = int(SR * slide_dur)
    freqs = np.ones(len(tt)) * target
    slide = freq * (target / freq) ** np.linspace(0, 1, slide_samples)
    freqs[:slide_samples] = slide
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    sig = np.sin(phase)
    return sig, [{"label": "portamento", "start": 0.0, "end": slide_dur}]


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


def gen_marcato(freq, dur):
    notes = RNG.integers(6, 10)
    return _gen_phrase(freq, dur, notes, duty=0.6, label="marcato")


def gen_tenuto(freq, dur):
    notes = RNG.integers(6, 10)
    return _gen_phrase(freq, dur, notes, duty=0.95, label="tenuto")


def gen_portato(freq, dur):
    notes = RNG.integers(6, 10)
    return _gen_phrase(freq, dur, notes, duty=0.6, label="portato")


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


def gen_rubato(freq, dur):
    notes = RNG.integers(8, 12)
    base = dur / notes
    iois = base + RNG.normal(0, base * 0.25, notes)
    iois = np.clip(iois, base * 0.4, base * 1.6)
    return _gen_tempo_phrase(freq, dur, notes, iois, "rubato")


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


# --- Timbre modification ---

def gen_muted(freq, dur):
    sig = sawtooth(freq, dur)
    sig = lowpass(sig, cutoff=freq * RNG.uniform(1.5, 2.5))
    return sig, [{"label": "muted", "start": 0.0, "end": dur}]


def gen_harmonics(freq, dur):
    """Play isolated harmonics in sequence."""
    quarter = dur / 4
    parts = []
    for k in [1, 2, 3, 4]:
        seg = sine(freq * k, quarter)
        fade = int(SR * 0.02)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        parts.append(seg)
    return np.concatenate(parts), [{"label": "harmonics", "start": 0.0, "end": dur}]


def gen_distortion(freq, dur):
    """Soft-clipped distortion with harmonic richness and post-EQ."""
    # Start with a harmonically rich signal
    sig = sawtooth(freq, dur)
    # Soft clip via tanh saturation
    gain = RNG.uniform(3, 8)
    sig = np.tanh(sig * gain)
    # Asymmetric clipping: positive side clips harder
    asym = RNG.uniform(0.0, 0.3)
    sig = np.where(sig > 0, sig * (1 - asym), sig)
    # Roll off harsh highs (cabinet sim)
    cutoff = freq * RNG.uniform(4, 8)
    cutoff = min(cutoff, SR / 2 - 1)
    sig = lowpass(sig, cutoff)
    return sig, [{"label": "distortion", "start": 0.0, "end": dur}]


def gen_palm_mute(freq, dur):
    sig = sawtooth(freq, dur)
    tt = t(dur)
    env = np.exp(-RNG.uniform(4, 8) * tt / dur)
    sig *= env
    sig = lowpass(sig, cutoff=freq * RNG.uniform(2, 4))
    return sig, [{"label": "palm_mute", "start": 0.0, "end": dur}]


# --- Articulation (new) ---

def gen_spiccato(freq, dur):
    """Bounced bow: short notes with sharp attack and fast decay."""
    notes = RNG.integers(8, 14)
    slot = dur / notes
    out = np.zeros(int(SR * dur))
    for i in range(notes):
        note_dur = slot * RNG.uniform(0.25, 0.4)
        start = int(SR * slot * i)
        seg = pick_waveform(freq, note_dur)
        # Sharp attack, exponential decay
        decay = np.exp(-8 * np.linspace(0, 1, len(seg)))
        seg *= decay
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out, [{"label": "spiccato", "start": 0.0, "end": dur}]


def gen_dead_note(freq, dur):
    """Muted percussive hit with no sustain."""
    tt = t(dur)
    # Noise burst with fast decay, minimal pitch content
    noise = RNG.normal(0, 1, len(tt))
    hit_dur = RNG.uniform(0.02, 0.05)
    hit_samples = int(SR * hit_dur)
    env = np.zeros(len(tt))
    env[:hit_samples] = np.exp(-20 * np.linspace(0, 1, hit_samples))
    sig = noise * env
    sig = lowpass(sig, freq * RNG.uniform(2, 4))
    return sig, [{"label": "dead_note", "start": 0.0, "end": hit_dur}]


# --- Ornaments (new) ---

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


def gen_flam(freq, dur):
    """Two near-simultaneous hits."""
    tt = t(dur)
    sig = np.zeros(len(tt))
    flam_gap = RNG.uniform(0.015, 0.04)
    # Scale hit duration with frequency
    min_hit_dur = max(4.0 / freq, 0.015)
    hit_dur = RNG.uniform(min_hit_dur, min_hit_dur * 2)
    noise_mix = np.clip(0.1 + (freq - 80) / 1000, 0.05, 0.4)
    decay_rate = np.clip(10 + (freq - 80) / 30, 8, 30)

    for offset, amp in [(0.0, 0.5), (flam_gap, 1.0)]:
        start = int(SR * offset)
        n_samp = int(SR * hit_dur)
        if start + n_samp > len(sig):
            break
        tt_hit = np.linspace(0, hit_dur, n_samp)
        hit = np.sin(2 * np.pi * freq * tt_hit)
        hit += RNG.normal(0, noise_mix, n_samp)
        hit *= np.exp(-decay_rate * np.linspace(0, 1, n_samp)) * amp
        sig[start:start + n_samp] += hit

    # Sustain rest of duration quietly
    tail_start = int(SR * (flam_gap + hit_dur))
    if tail_start < len(sig):
        tail = pick_waveform(freq, dur - flam_gap - hit_dur) * 0.15
        end = tail_start + len(tail)
        if end > len(sig):
            tail = tail[:len(sig) - tail_start]
            end = len(sig)
        sig[tail_start:end] = tail
    return sig, [{"label": "flam", "start": 0.0, "end": round(flam_gap + hit_dur, 4)}]


def gen_ghost_note(freq, dur):
    """Very quiet, muffled notes within a pattern."""
    notes = RNG.integers(6, 10)
    slot = dur / notes
    out = np.zeros(int(SR * dur))
    ghost_indices = set(RNG.choice(notes, size=max(1, notes // 2), replace=False))
    for i in range(notes):
        start = int(SR * slot * i)
        seg = pick_waveform(freq, slot * 0.5)
        fade = int(SR * 0.008)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        if i in ghost_indices:
            seg *= RNG.uniform(0.05, 0.15)
            seg = lowpass(seg, freq * 2)
        else:
            seg *= 0.7
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out, [{"label": "ghost_note", "start": 0.0, "end": dur}]


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


# --- String / plucked ---

def gen_pizzicato(freq, dur):
    """Plucked string: sharp attack, fast decay, rich harmonics."""
    tt = t(dur)
    # Multiple harmonics for richness
    sig = np.zeros(len(tt))
    for k in range(1, 6):
        amp = 1.0 / k
        sig += amp * np.sin(2 * np.pi * freq * k * tt)
    # Fast exponential decay
    decay_rate = RNG.uniform(6, 12)
    env = np.exp(-decay_rate * tt / dur)
    sig *= env
    return sig, [{"label": "pizzicato", "start": 0.0, "end": dur}]


def gen_hammer_on(freq, dur):
    """Hammer-on: second note sounds without a new attack."""
    tt = t(dur)
    interval = RNG.choice([2, 3, 4, 5])
    freq2 = freq * (2 ** (interval / 12))
    mid = len(tt) // 2
    # First note: normal attack + decay
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[mid:] = freq2
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    # Smooth transition (no re-attack)
    env = np.ones(len(tt))
    env[:mid] = np.exp(-2 * np.linspace(0, 1, mid))
    env[mid:] = env[mid - 1] * np.exp(-3 * np.linspace(0, 1, len(tt) - mid))
    # Small bump at hammer point
    bump_len = int(SR * 0.01)
    if mid + bump_len < len(env):
        env[mid:mid + bump_len] *= 1.3
    sig *= env
    hammer_time = mid / SR
    return sig, [{"label": "hammer_on", "start": round(hammer_time, 4),
                  "end": round(hammer_time + 0.05, 4)}]


def gen_slide(freq, dur):
    """Short, percussive pitch slide between two notes."""
    tt = t(dur)
    interval = RNG.choice([-7, -5, -3, 3, 5, 7])
    freq2 = freq * (2 ** (interval / 12))
    slide_dur = RNG.uniform(0.05, 0.15)
    slide_start = RNG.uniform(0.2, 0.5) * dur
    slide_samples = int(SR * slide_dur)
    slide_start_samp = int(SR * slide_start)
    slide_end_samp = slide_start_samp + slide_samples
    inst_freq = np.ones(len(tt)) * freq
    if slide_end_samp < len(tt):
        inst_freq[slide_start_samp:slide_end_samp] = np.linspace(freq, freq2, slide_samples)
        inst_freq[slide_end_samp:] = freq2
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    return sig, [{"label": "slide", "start": round(slide_start, 4),
                  "end": round(slide_start + slide_dur, 4)}]


# --- Wind ---

def gen_flutter(freq, dur):
    """Flutter tongue: rapid amplitude + pitch modulation."""
    tt = t(dur)
    flutter_rate = RNG.uniform(20, 35)
    # Pitch flutter
    inst_freq = freq + RNG.uniform(2, 5) * np.sin(2 * np.pi * flutter_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    # Amplitude flutter
    amp_mod = 0.7 + 0.3 * np.sin(2 * np.pi * flutter_rate * tt)
    sig *= amp_mod
    return sig, [{"label": "flutter", "start": 0.0, "end": dur}]


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
        gen_crescendo, gen_decrescendo, gen_sforzando, gen_swell,
        gen_fortepiano, gen_subito_forte, gen_subito_piano, gen_morendo,
    ],
    "ornaments": [
        gen_vibrato, gen_tremolo, gen_trill, gen_glissando,
        gen_mordent, gen_grace_note, gen_pitch_bend, gen_turn, gen_portamento,
        gen_scoop, gen_fall_off,
    ],
    "articulation": [
        gen_staccato, gen_legato, gen_marcato, gen_tenuto, gen_portato, gen_accent,
        gen_spiccato, gen_dead_note,
    ],
    "tempo": [
        gen_accelerando, gen_ritardando, gen_rubato, gen_fermata, gen_caesura,
    ],
    "timbre": [
        gen_muted, gen_distortion, gen_palm_mute, gen_harmonics,
    ],
    "percussion": [
        gen_roll, gen_flam, gen_ghost_note, gen_choke,
    ],
    "string": [
        gen_pizzicato, gen_hammer_on, gen_slide,
    ],
    "wind": [
        gen_flutter, gen_arpeggio,
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

def overlay_morendo(sig, dur):
    env = np.exp(-RNG.uniform(2, 5) * np.linspace(0, 1, len(sig)))
    return sig * env, {"label": "morendo", "start": 0.0, "end": dur}

def overlay_tremolo(sig, dur):
    tt = np.linspace(0, dur, len(sig))
    rate = RNG.uniform(5, 10)
    depth = RNG.uniform(0.5, 0.9)
    env = 1.0 - depth * 0.5 * (1 + np.sin(2 * np.pi * rate * tt))
    return sig * env, {"label": "tremolo", "start": 0.0, "end": dur}

def overlay_vibrato(sig, dur, freq):
    tt = t(dur)
    n = min(len(tt), len(sig))
    vib_rate = RNG.uniform(4, 7)
    vib_depth = RNG.uniform(3, 15)
    inst_freq = freq + vib_depth * np.sin(2 * np.pi * vib_rate * tt[:n])
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase), {"label": "vibrato", "start": 0.0, "end": dur}

def overlay_distortion(sig, dur):
    gain = RNG.uniform(3, 8)
    return np.tanh(sig * gain), {"label": "distortion", "start": 0.0, "end": dur}

def overlay_muted(sig, dur, freq):
    cutoff = freq * RNG.uniform(1.5, 2.5)
    return lowpass(sig, cutoff), {"label": "muted", "start": 0.0, "end": dur}

# Each overlay tagged with a kind so we know which arg signature to use.
# "amp" = (sig, dur), "sig" = (sig, dur), "freq" = (sig, dur, freq)
ALL_OVERLAYS = [
    ("amp",  overlay_crescendo),
    ("amp",  overlay_decrescendo),
    ("amp",  overlay_swell),
    ("amp",  overlay_morendo),
    ("amp",  overlay_tremolo),
    ("sig",  overlay_distortion),
    ("freq", overlay_vibrato),
    ("freq", overlay_muted),
]

# Which overlay kinds are compatible with each base category.
COMPATIBLE_KINDS = {
    "ornaments":     {"amp", "sig"},
    "timbre":        {"amp"},
    "articulation":  {"amp", "freq"},
    "tempo":         {"amp", "freq"},
    "dynamics":      {"sig", "freq"},
    "percussion":    {"amp"},
    "string":        {"amp", "sig"},
    "wind":          {"amp"},
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

def generate_sample(idx, n_total):
    """Generate one training sample with randomized parameters.
    Tracks label counts and skips overlays that are overrepresented."""
    freq = RNG.uniform(60, 1200)
    dur = RNG.uniform(1.5, 4.0)

    if RNG.random() < 0.5:
        sig, anns = generate_single(freq, dur)
    else:
        sig, anns = generate_composed(freq, dur)

    # Cap overlay labels: ideal count is n_total / num_labels.
    # Drop overlay annotations (not the base) if they exceed 1.5x the ideal.
    ideal = n_total / len(ALL_GENERATORS)
    cap = ideal * 1.5
    if len(anns) > 1:
        filtered = [anns[0]]
        for ann in anns[1:]:
            if _label_counts.get(ann["label"], 0) < cap:
                filtered.append(ann)
        anns = filtered

    for ann in anns:
        _label_counts[ann["label"]] = _label_counts.get(ann["label"], 0) + 1

    snr = RNG.uniform(20, 50)
    sig = add_noise(sig, snr_db=snr)
    sig = normalize(sig)

    for ann in anns:
        ann["start"] = round(ann["start"], 4)
        ann["end"] = round(min(ann["end"], dur), 4)

    return sig, anns, dur


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
    args = parser.parse_args()

    global RNG, _label_counts, _gen_cycle_idx
    RNG = np.random.default_rng(args.seed)
    _label_counts = {}
    _gen_cycle_idx = 0

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