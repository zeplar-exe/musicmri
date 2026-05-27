"""
Batch 2: Additional expressive technique demonstrations.
Builds on the same structure as synth_techniques.py.
"""

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import os

SR = 44100

def t(duration):
    return np.linspace(0, duration, int(SR * duration), endpoint=False)


def sine(freq, duration):
    return np.sin(2 * np.pi * freq * t(duration))


def sawtooth(freq, duration, n_harmonics=15):
    """Sawtooth wave via additive synthesis. Harmonically rich for filtering."""
    tt = t(duration)
    out = np.zeros(len(tt))
    for k in range(1, n_harmonics + 1):
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * tt) / k
    return out * (2 / np.pi)


def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


def save(name, signal, out_dir="/mnt/user-data/outputs/techniques"):
    os.makedirs(out_dir, exist_ok=True)
    signal = normalize(signal)
    pcm = (signal * 32767 * 0.8).astype(np.int16)
    path = os.path.join(out_dir, f"{name}.wav")
    write(path, SR, pcm)
    print(f"  wrote {path}")


# =====================
# DYNAMICS (additional)
# =====================

def fortepiano(freq, duration):
    """Loud attack, immediate drop to soft sustain."""
    tt = t(duration)
    envelope = np.ones(len(tt)) * 0.15
    # Sharp attack: first 5% at full volume, fast drop
    attack_end = int(len(tt) * 0.05)
    envelope[:attack_end] = 1.0
    drop_len = int(len(tt) * 0.08)
    envelope[attack_end:attack_end + drop_len] = np.linspace(1.0, 0.15, drop_len)
    return envelope * np.sin(2 * np.pi * freq * tt)


def subito_forte(freq, duration):
    """Playing soft, then sudden jump to loud."""
    tt = t(duration)
    envelope = np.ones(len(tt)) * 0.15
    mid = len(tt) // 2
    envelope[mid:] = 1.0
    return envelope * np.sin(2 * np.pi * freq * tt)


def subito_piano(freq, duration):
    """Playing loud, then sudden drop to soft."""
    tt = t(duration)
    envelope = np.ones(len(tt)) * 1.0
    mid = len(tt) // 2
    envelope[mid:] = 0.15
    return envelope * np.sin(2 * np.pi * freq * tt)


def morendo(freq, duration):
    """Dying away: volume decreases AND tempo slightly slows.
    We simulate by stretching note spacing + amplitude fade."""
    tt = t(duration)
    # Exponential amplitude decay
    envelope = np.exp(-3 * tt / duration)
    return envelope * np.sin(2 * np.pi * freq * tt)


# ==========================
# ARTICULATION (additional)
# ==========================

def _phrase(freq, duration, notes, duty, attack_sharpness=0.01):
    """Generic phrase generator.
    duty: fraction of each slot filled by the note.
    attack_sharpness: seconds for fade-in.
    """
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    note_dur = slot * duty
    for i in range(notes):
        start = int(SR * slot * i)
        seg = sine(freq, note_dur)
        # Attack
        fade_in_len = min(int(SR * attack_sharpness), len(seg) // 4)
        if fade_in_len > 0:
            seg[:fade_in_len] *= np.linspace(0, 1, fade_in_len)
        # Release
        fade_out_len = min(int(SR * 0.02), len(seg) // 4)
        seg[-fade_out_len:] *= np.linspace(1, 0, fade_out_len)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] += seg
    return out


def marcato_phrase(freq, duration, notes=8):
    """Strongly accented notes: sharp attack, ~60% duty."""
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    note_dur = slot * 0.6
    for i in range(notes):
        start = int(SR * slot * i)
        seg = sine(freq, note_dur)
        # Sharp attack with initial amplitude boost
        boost_len = int(SR * 0.03)
        boost_env = np.ones(len(seg))
        boost_env[:boost_len] = np.linspace(1.5, 1.0, boost_len)
        seg *= boost_env
        # Release
        fade_len = min(int(SR * 0.02), len(seg) // 4)
        seg[-fade_len:] *= np.linspace(1, 0, fade_len)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] += seg
    return out


def tenuto_phrase(freq, duration, notes=8):
    """Full duration, slight emphasis. Duty ~95%."""
    return _phrase(freq, duration, notes, duty=0.95, attack_sharpness=0.005)


def portato_phrase(freq, duration, notes=8):
    """Between staccato and legato. Duty ~60%, gentle attack."""
    return _phrase(freq, duration, notes, duty=0.6, attack_sharpness=0.01)


def accent_phrase(freq, duration, notes=8):
    """Single accented note in an otherwise uniform phrase."""
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    accent_idx = 3  # accent the 4th note
    for i in range(notes):
        start = int(SR * slot * i)
        seg = sine(freq, slot * 0.7)
        amp = 1.5 if i == accent_idx else 0.5
        seg *= amp
        fade_len = min(int(SR * 0.02), len(seg) // 4)
        seg[-fade_len:] *= np.linspace(1, 0, fade_len)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] += seg
    return out


# =======================
# ORNAMENTS (additional)
# =======================

def turn(freq, duration):
    """Four-note ornament: above - main - below - main, then sustain."""
    tt = t(duration)
    above = freq * (2 ** (2 / 12))   # whole step up
    below = freq / (2 ** (2 / 12))   # whole step down

    ornament_dur = 0.20  # total ornament time
    orn_samples = int(SR * ornament_dur)
    seg = orn_samples // 4

    inst_freq = np.ones(len(tt)) * freq
    inst_freq[0:seg] = above
    inst_freq[seg:2*seg] = freq
    inst_freq[2*seg:3*seg] = below
    inst_freq[3*seg:4*seg] = freq
    # Rest is already freq

    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def portamento(freq_start, freq_end, duration):
    """Short smooth pitch slide, then sustain at target.
    Distinguished from glissando by being shorter and landing on the target quickly."""
    tt = t(duration)
    slide_dur = 0.25  # seconds
    slide_samples = int(SR * slide_dur)

    freqs = np.ones(len(tt)) * freq_end
    # Exponential slide in the first portion
    slide = freq_start * (freq_end / freq_start) ** (np.linspace(0, 1, slide_samples))
    freqs[:slide_samples] = slide

    phase = 2 * np.pi * np.cumsum(freqs) / SR
    return np.sin(phase)


# ==============================
# TEMPO MODIFICATION (additional)
# ==============================

def rubato_phrase(freq, duration, notes=10):
    """Flexible timing: IOIs randomly perturbed around a mean."""
    rng = np.random.default_rng(42)
    base_ioi = duration / notes
    iois = base_ioi + rng.normal(0, base_ioi * 0.25, notes)
    iois = np.clip(iois, base_ioi * 0.4, base_ioi * 1.6)
    iois = iois / iois.sum() * duration  # normalize

    out = np.zeros(int(SR * duration))
    pos = 0.0
    for ioi in iois:
        note_dur = ioi * 0.7
        start = int(SR * pos)
        seg = sine(freq, note_dur)
        fade = np.linspace(1, 0, len(seg)) ** 2
        seg *= fade
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out


def fermata_phrase(freq, duration, notes=6):
    """Regular phrase with one note held much longer than the rest."""
    # 6 notes, the 3rd one gets ~40% of total duration
    iois = np.ones(notes)
    fermata_idx = 2
    iois[fermata_idx] = 4.0  # 4x longer
    iois = iois / iois.sum() * duration

    out = np.zeros(int(SR * duration))
    pos = 0.0
    for i, ioi in enumerate(iois):
        note_dur = ioi * 0.85
        start = int(SR * pos)
        seg = sine(freq, note_dur)
        fade_len = min(int(SR * 0.03), len(seg) // 4)
        seg[-fade_len:] *= np.linspace(1, 0, fade_len)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out


def caesura_phrase(freq, duration, notes=8):
    """Phrase with a full stop (silence) in the middle."""
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    pause_at = 4  # silence instead of 5th note
    for i in range(notes):
        if i == pause_at:
            continue  # gap
        start = int(SR * slot * i)
        seg = sine(freq, slot * 0.7)
        fade_len = min(int(SR * 0.02), len(seg) // 4)
        seg[-fade_len:] *= np.linspace(1, 0, fade_len)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out


# ==============================
# TIMBRE MODIFICATION
# ==============================

def lowpass(signal, cutoff, sr=SR, order=4):
    """Butterworth low-pass filter."""
    sos = butter(order, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, signal)


def bandpass(signal, low, high, sr=SR, order=2):
    """Butterworth band-pass filter."""
    sos = butter(order, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, signal)


def muted(freq, duration):
    """Con sordino: low-pass filtered to remove upper harmonics."""
    sig = sawtooth(freq, duration)
    return lowpass(sig, cutoff=freq * 2)


def harmonics_demo(freq, duration):
    """Isolated harmonics: play fundamental, then 2nd, 3rd, 4th harmonic."""
    quarter = duration / 4
    parts = []
    for k in [1, 2, 3, 4]:
        seg = sine(freq * k, quarter)
        # Fade in/out
        fade = int(SR * 0.02)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        parts.append(seg)
    return np.concatenate(parts)


def wah_effect(freq, duration, wah_rate=2.5):
    """Swept bandpass filter (wah pedal) on a harmonically rich tone."""
    sig = sawtooth(freq, duration)
    tt = t(duration)
    out = np.zeros(len(tt))
    # Chunk-based processing: sweep center frequency
    chunk_size = 512
    n_chunks = len(tt) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        # Sweep center freq between freq*1.5 and freq*8
        phase = 2 * np.pi * wah_rate * (start / SR)
        center = freq * 1.5 + (freq * 6) * (0.5 + 0.5 * np.sin(phase))
        bw = center * 0.4
        lo = max(center - bw, 20)
        hi = min(center + bw, SR / 2 - 1)
        if hi <= lo:
            hi = lo + 100
        chunk = sig[start:end]
        try:
            out[start:end] = bandpass(chunk, lo, hi)
        except Exception:
            out[start:end] = chunk
    # Handle remainder
    remainder = len(tt) - n_chunks * chunk_size
    if remainder > 0:
        out[-remainder:] = sig[-remainder:]
    return out


def distortion(freq, duration, gain=8, threshold=0.3):
    """Hard clipping distortion on a sine wave."""
    sig = sine(freq, duration) * gain
    return np.clip(sig, -threshold, threshold)


def palm_mute(freq, duration):
    """Damped string: fast amplitude decay + low-pass."""
    sig = sawtooth(freq, duration)
    tt = t(duration)
    # Quick exponential decay per "note"
    envelope = np.exp(-6 * tt / duration)
    sig *= envelope
    return lowpass(sig, cutoff=freq * 3)


# =====================
# GENERATE ALL
# =====================

if __name__ == "__main__":
    F = 440
    D = 2.0

    print("Generating dynamics (batch 2)...")
    save("16_fortepiano", fortepiano(F, D))
    save("17_subito_forte", subito_forte(F, D))
    save("18_subito_piano", subito_piano(F, D))
    save("19_morendo", morendo(F, D))

    print("Generating articulations (batch 2)...")
    save("20_marcato", marcato_phrase(F, D))
    save("21_tenuto", tenuto_phrase(F, D))
    save("22_portato", portato_phrase(F, D))
    save("23_accent", accent_phrase(F, D))

    print("Generating ornaments (batch 2)...")
    save("24_turn", turn(F, D))
    save("25_portamento", portamento(F, F * 1.5, D))  # A4 to E5

    print("Generating tempo modifications (batch 2)...")
    save("26_rubato", rubato_phrase(F, D))
    save("27_fermata", fermata_phrase(F, D))
    save("28_caesura", caesura_phrase(F, D))

    print("Generating timbre modifications...")
    save("29_muted", muted(F, D))
    save("30_harmonics", harmonics_demo(F, D))
    save("31_wah", wah_effect(F, D))
    save("32_distortion", distortion(F, D))
    save("33_palm_mute", palm_mute(F, D))

    print("Done. Batch 2 complete.")
