"""
Synthesize short audio demonstrations of musical expressive techniques.
Each function takes a base frequency and duration and returns a numpy array of samples.
"""

import numpy as np
from scipy.io.wavfile import write
import os

SR = 44100  # sample rate

def t(duration):
    """Time array."""
    return np.linspace(0, duration, int(SR * duration), endpoint=False)


def sine(freq, duration):
    """Plain sine tone for reference."""
    return np.sin(2 * np.pi * freq * t(duration))


# --- Dynamics ---

def crescendo(freq, duration):
    """Linear amplitude ramp from ~0 to 1."""
    tt = t(duration)
    envelope = np.linspace(0.05, 1.0, len(tt))
    return envelope * np.sin(2 * np.pi * freq * tt)


def decrescendo(freq, duration):
    """Linear amplitude ramp from 1 to ~0."""
    tt = t(duration)
    envelope = np.linspace(1.0, 0.05, len(tt))
    return envelope * np.sin(2 * np.pi * freq * tt)


def sforzando(freq, duration):
    """Sharp accent at the start, then soft sustain."""
    tt = t(duration)
    # Fast exponential decay from 1.0 to 0.15 in the first 10%, then sustain
    envelope = np.ones(len(tt)) * 0.15
    attack_len = int(len(tt) * 0.10)
    envelope[:attack_len] = 1.0 * np.exp(-5 * np.linspace(0, 1, attack_len))
    return envelope * np.sin(2 * np.pi * freq * tt)


def swell(freq, duration):
    """Crescendo then decrescendo (diamond shape)."""
    tt = t(duration)
    mid = len(tt) // 2
    envelope = np.concatenate([
        np.linspace(0.05, 1.0, mid),
        np.linspace(1.0, 0.05, len(tt) - mid)
    ])
    return envelope * np.sin(2 * np.pi * freq * tt)


# --- Ornaments ---

def vibrato(freq, duration, vib_rate=5.5, vib_depth=8):
    """Frequency modulation: periodic pitch oscillation.
    vib_rate: Hz of the vibrato oscillation
    vib_depth: max deviation in Hz
    """
    tt = t(duration)
    # Integrate instantaneous frequency to get phase
    inst_freq = freq + vib_depth * np.sin(2 * np.pi * vib_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def tremolo(freq, duration, trem_rate=7, trem_depth=0.8):
    """Amplitude modulation: periodic volume oscillation."""
    tt = t(duration)
    envelope = 1.0 - trem_depth * 0.5 * (1 + np.sin(2 * np.pi * trem_rate * tt))
    return envelope * np.sin(2 * np.pi * freq * tt)


def trill(freq, duration, interval=2):
    """Rapid alternation between two pitches.
    interval: semitones above the base note
    """
    tt = t(duration)
    freq_upper = freq * (2 ** (interval / 12))
    # 8 Hz square wave to switch between pitches
    switch = (np.sin(2 * np.pi * 8 * tt) > 0).astype(float)
    inst_freq = freq * (1 - switch) + freq_upper * switch
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def glissando(freq_start, freq_end, duration):
    """Continuous pitch slide between two frequencies."""
    tt = t(duration)
    # Exponential interpolation (perceptually linear pitch)
    freqs = freq_start * (freq_end / freq_start) ** (tt / duration)
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    return np.sin(phase)


def mordent(freq, duration):
    """Quick dip to lower neighbor and back at the note onset."""
    tt = t(duration)
    lower = freq / (2 ** (2 / 12))  # whole step below
    mordent_dur = 0.08  # seconds for the ornament
    mordent_samples = int(SR * mordent_dur)

    # Three segments: main -> lower -> main, each 1/3 of mordent_dur
    seg = mordent_samples // 3
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[seg:2*seg] = lower

    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def grace_note(freq, duration):
    """Short note a step below, then the main note."""
    tt = t(duration)
    grace_freq = freq / (2 ** (2 / 12))
    grace_samples = int(SR * 0.06)
    inst_freq = np.ones(len(tt)) * freq
    inst_freq[:grace_samples] = grace_freq
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def pitch_bend(freq, duration, bend_semitones=2):
    """Pitch pushed sharp then released back, guitar-style."""
    tt = t(duration)
    bend_freq = freq * (2 ** (bend_semitones / 12))
    # Ramp up to bent pitch over first 30%, hold 40%, release 30%
    n = len(tt)
    ramp_up = int(n * 0.3)
    hold = int(n * 0.4)
    ramp_down = n - ramp_up - hold
    freq_env = np.concatenate([
        np.linspace(freq, bend_freq, ramp_up),
        np.ones(hold) * bend_freq,
        np.linspace(bend_freq, freq, ramp_down)
    ])
    phase = 2 * np.pi * np.cumsum(freq_env) / SR
    return np.sin(phase)


# --- Articulation ---

def staccato_phrase(freq, duration, notes=8):
    """Short detached notes. Note duration = 30% of slot."""
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    note_len = int(SR * slot * 0.3)
    for i in range(notes):
        start = int(SR * slot * i)
        end = start + note_len
        if end > len(out):
            break
        segment = np.sin(2 * np.pi * freq * t(slot * 0.3))
        # Apply quick fade-out
        fade = np.linspace(1, 0, len(segment)) ** 2
        out[start:start+len(segment)] = segment * fade
    return out


def legato_phrase(freq, duration, notes=8):
    """Smooth connected notes. Note duration = 100% of slot."""
    slot = duration / notes
    out = np.zeros(int(SR * duration))
    note_len = int(SR * slot)
    for i in range(notes):
        start = int(SR * slot * i)
        segment = np.sin(2 * np.pi * freq * t(slot))
        # Gentle crossfade
        fade_len = min(int(SR * 0.02), len(segment) // 4)
        segment[:fade_len] *= np.linspace(0, 1, fade_len)
        segment[-fade_len:] *= np.linspace(1, 0, fade_len)
        end = start + len(segment)
        if end > len(out):
            segment = segment[:len(out) - start]
            end = len(out)
        out[start:end] += segment
    return out


# --- Tempo Modification ---

def accelerando_phrase(freq, duration, notes=10):
    """Notes that get progressively closer together."""
    # IOIs decrease linearly
    iois = np.linspace(1.0, 0.3, notes)
    iois = iois / iois.sum() * duration  # normalize to fill duration
    out = np.zeros(int(SR * duration))
    pos = 0.0
    for ioi in iois:
        note_dur = ioi * 0.7
        start = int(SR * pos)
        seg = np.sin(2 * np.pi * freq * t(note_dur))
        fade = np.linspace(1, 0, len(seg)) ** 2
        seg *= fade
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out


def ritardando_phrase(freq, duration, notes=10):
    """Notes that get progressively farther apart."""
    iois = np.linspace(0.3, 1.0, notes)
    iois = iois / iois.sum() * duration
    out = np.zeros(int(SR * duration))
    pos = 0.0
    for ioi in iois:
        note_dur = ioi * 0.7
        start = int(SR * pos)
        seg = np.sin(2 * np.pi * freq * t(note_dur))
        fade = np.linspace(1, 0, len(seg)) ** 2
        seg *= fade
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
        pos += ioi
    return out


# --- Generate all ---

def normalize(signal):
    """Normalize to [-1, 1] range."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


def save(name, signal, out_dir="/mnt/user-data/outputs/techniques"):
    os.makedirs(out_dir, exist_ok=True)
    signal = normalize(signal)
    # Convert to 16-bit PCM
    pcm = (signal * 32767 * 0.8).astype(np.int16)
    path = os.path.join(out_dir, f"{name}.wav")
    write(path, SR, pcm)
    print(f"  wrote {path}")


if __name__ == "__main__":
    F = 440  # A4 as base
    D = 2.0  # 2 seconds default

    print("Generating reference tone...")
    save("00_reference_tone", sine(F, D))

    print("Generating dynamics...")
    save("01_crescendo", crescendo(F, D))
    save("02_decrescendo", decrescendo(F, D))
    save("03_sforzando", sforzando(F, D))
    save("04_swell", swell(F, D))

    print("Generating ornaments...")
    save("05_vibrato", vibrato(F, D))
    save("06_tremolo", tremolo(F, D))
    save("07_trill", trill(F, D))
    save("08_glissando", glissando(F, F * 2, D))  # A4 to A5
    save("09_mordent", mordent(F, D))
    save("10_grace_note", grace_note(F, D))
    save("11_pitch_bend", pitch_bend(F, D))

    print("Generating articulations...")
    save("12_staccato", staccato_phrase(F, D))
    save("13_legato", legato_phrase(F, D))

    print("Generating tempo modifications...")
    save("14_accelerando", accelerando_phrase(F, D))
    save("15_ritardando", ritardando_phrase(F, D))

    print("Done.")
