"""
Batch 3: Reference demos for the 13 new technique labels,
plus a regenerated distortion with soft clipping.
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
    tt = t(duration)
    out = np.zeros(len(tt))
    for k in range(1, n_harmonics + 1):
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * tt) / k
    return out * (2 / np.pi)

def lowpass(signal, cutoff, order=4):
    sos = butter(order, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, signal)

def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return signal
    return signal / peak

def save(name, signal, idx):
    signal = normalize(signal) * 0.8
    pcm = (signal * 32767).astype(np.int16)
    path = os.path.join(os.path.dirname(__file__), f"{idx:02d}_{name}.wav")
    write(path, SR, pcm)
    print(f"  {path}")


# ============================================================

FREQ = 440
DUR = 2.5


def spiccato():
    """Bounced bow: short notes with sharp attack and fast decay."""
    notes = 10
    slot = DUR / notes
    out = np.zeros(int(SR * DUR))
    for i in range(notes):
        note_dur = slot * 0.3
        start = int(SR * slot * i)
        seg = sine(FREQ, note_dur)
        decay = np.exp(-8 * np.linspace(0, 1, len(seg)))
        seg *= decay
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out


def dead_note():
    """Muted percussive hit, no pitch sustain."""
    tt = t(DUR)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(tt))
    hit_samples = int(SR * 0.04)
    env = np.zeros(len(tt))
    env[:hit_samples] = np.exp(-20 * np.linspace(0, 1, hit_samples))
    sig = noise * env
    sig = lowpass(sig, FREQ * 3)
    return sig


def scoop():
    """Approach a note from below."""
    tt = t(DUR)
    scoop_dur = 0.1
    scoop_samples = int(SR * scoop_dur)
    start_freq = FREQ / (2 ** (3 / 12))
    inst_freq = np.ones(len(tt)) * FREQ
    inst_freq[:scoop_samples] = np.linspace(start_freq, FREQ, scoop_samples)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def fall_off():
    """Pitch drops off at the end of a note."""
    tt = t(DUR)
    fall_dur = 0.15
    fall_samples = int(SR * fall_dur)
    end_freq = FREQ / (2 ** (5 / 12))
    inst_freq = np.ones(len(tt)) * FREQ
    fall_start = len(tt) - fall_samples
    inst_freq[fall_start:] = np.linspace(FREQ, end_freq, fall_samples)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    env = np.ones(len(tt))
    env[fall_start:] = np.linspace(1.0, 0.1, fall_samples)
    return sig * env


def roll():
    """Rapid repeated strikes (drum roll)."""
    rate = 20  # hits per second
    n_hits = int(rate * DUR)
    out = np.zeros(int(SR * DUR))
    hit_dur = 0.02
    rng = np.random.default_rng(1)
    for i in range(n_hits):
        pos = i / rate
        start = int(SR * pos)
        n_samp = int(SR * hit_dur)
        if start + n_samp > len(out):
            break
        hit = rng.normal(0, 1, n_samp) * 0.3 + np.sin(2 * np.pi * FREQ * np.linspace(0, hit_dur, n_samp)) * 0.7
        hit *= np.exp(-30 * np.linspace(0, 1, n_samp))
        hit *= rng.uniform(0.6, 1.0)
        out[start:start + n_samp] += hit
    return out


def flam():
    """Two near-simultaneous hits."""
    tt = t(DUR)
    sig = np.zeros(len(tt))
    flam_gap = 0.025
    hit_dur = 0.03
    rng = np.random.default_rng(2)
    for offset, amp in [(0.0, 0.5), (flam_gap, 1.0)]:
        start = int(SR * offset)
        n_samp = int(SR * hit_dur)
        if start + n_samp > len(sig):
            break
        hit = np.sin(2 * np.pi * FREQ * np.linspace(0, hit_dur, n_samp))
        hit += rng.normal(0, 0.3, n_samp)
        hit *= np.exp(-25 * np.linspace(0, 1, n_samp)) * amp
        sig[start:start + n_samp] += hit
    # Sustain
    tail_start = int(SR * (flam_gap + hit_dur))
    if tail_start < len(sig):
        tail = sine(FREQ, DUR - flam_gap - hit_dur) * 0.15
        sig[tail_start:tail_start + len(tail)] = tail[:len(sig) - tail_start]
    return sig


def ghost_note():
    """Very quiet muffled notes within a louder pattern."""
    notes = 8
    slot = DUR / notes
    out = np.zeros(int(SR * DUR))
    ghosts = {1, 3, 5, 7}
    for i in range(notes):
        start = int(SR * slot * i)
        seg = sine(FREQ, slot * 0.5)
        fade = int(SR * 0.008)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        if i in ghosts:
            seg *= 0.1
            seg = lowpass(seg, FREQ * 2)
        else:
            seg *= 0.7
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out


def choke():
    """Abrupt muting of a resonant sound."""
    tt = t(DUR)
    sig = sine(FREQ, DUR)
    sig += 0.3 * np.sin(2 * np.pi * FREQ * 2.01 * tt)
    ring_samples = int(SR * 0.4 * DUR)
    cut_samples = int(SR * 0.02)
    env = np.ones(len(tt))
    cut_end = min(ring_samples + cut_samples, len(tt))
    env[ring_samples:cut_end] = np.linspace(1.0, 0.0, cut_end - ring_samples)
    env[cut_end:] = 0.0
    return sig * env


def pizzicato():
    """Plucked string: sharp attack, fast decay, rich harmonics."""
    tt = t(DUR)
    sig = np.zeros(len(tt))
    for k in range(1, 6):
        sig += (1.0 / k) * np.sin(2 * np.pi * FREQ * k * tt)
    env = np.exp(-8 * tt / DUR)
    return sig * env


def hammer_on():
    """Second note sounds without a new attack."""
    tt = t(DUR)
    freq2 = FREQ * (2 ** (4 / 12))  # major third up
    mid = len(tt) // 2
    inst_freq = np.ones(len(tt)) * FREQ
    inst_freq[mid:] = freq2
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    env = np.ones(len(tt))
    env[:mid] = np.exp(-2 * np.linspace(0, 1, mid))
    env[mid:] = env[mid - 1] * np.exp(-3 * np.linspace(0, 1, len(tt) - mid))
    bump_len = int(SR * 0.01)
    if mid + bump_len < len(env):
        env[mid:mid + bump_len] *= 1.3
    return sig * env


def slide():
    """Short pitch slide between two notes."""
    tt = t(DUR)
    freq2 = FREQ * (2 ** (5 / 12))
    slide_dur = 0.1
    slide_start = 0.4 * DUR
    slide_samples = int(SR * slide_dur)
    slide_start_samp = int(SR * slide_start)
    slide_end_samp = slide_start_samp + slide_samples
    inst_freq = np.ones(len(tt)) * FREQ
    inst_freq[slide_start_samp:slide_end_samp] = np.linspace(FREQ, freq2, slide_samples)
    inst_freq[slide_end_samp:] = freq2
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    return np.sin(phase)


def flutter():
    """Flutter tongue: rapid amplitude + pitch modulation."""
    tt = t(DUR)
    flutter_rate = 25
    inst_freq = FREQ + 3 * np.sin(2 * np.pi * flutter_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    amp_mod = 0.7 + 0.3 * np.sin(2 * np.pi * flutter_rate * tt)
    return sig * amp_mod


def arpeggio():
    """Broken major chord played in sequence."""
    intervals = [0, 4, 7, 12]
    note_dur = DUR / len(intervals)
    out = np.zeros(int(SR * DUR))
    for i, semi in enumerate(intervals):
        note_freq = FREQ * (2 ** (semi / 12))
        start = int(SR * note_dur * i)
        seg = sine(note_freq, note_dur)
        decay = np.exp(-3 * np.linspace(0, 1, len(seg)))
        seg *= decay
        fade = int(SR * 0.005)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out


def distortion_fixed():
    """Soft-clipped distortion with tanh saturation and post-EQ."""
    sig = sawtooth(FREQ, DUR)
    sig = np.tanh(sig * 5)
    sig = np.where(sig > 0, sig * 0.85, sig)
    sig = lowpass(sig, FREQ * 6)
    return sig


# ============================================================

if __name__ == "__main__":
    print("Generating batch 3 reference sounds...")

    # Regenerate distortion with the fixed version
    save("distortion", distortion_fixed(), 32)

    # New techniques starting at 34
    demos = [
        ("spiccato",    spiccato,    34),
        ("dead_note",   dead_note,   35),
        ("scoop",       scoop,       36),
        ("fall_off",    fall_off,    37),
        ("roll",        roll,        38),
        ("flam",        flam,        39),
        ("ghost_note",  ghost_note,  40),
        ("choke",       choke,       41),
        ("pizzicato",   pizzicato,   42),
        ("hammer_on",   hammer_on,   43),
        ("slide",       slide,       44),
        ("flutter",     flutter,     45),
        ("arpeggio",    arpeggio,    46),
    ]

    for name, fn, idx in demos:
        save(name, fn(), idx)

    print("Done.")
