"""
Synth song builder — sequences technique generators into listenable music.

Usage:
    python synth_song.py                          # default demo
    python synth_song.py --bpm 90 --key C minor   # custom key/tempo
    python synth_song.py --out my_track.wav
"""

import argparse
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt

SR = 44100

# ============================================================
# Musical primitives
# ============================================================

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":     [0, 2, 4, 7, 9],
    "minor_pent":     [0, 3, 5, 7, 10],
    "blues":          [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}


def note_freq(note_name, octave=4):
    """Get frequency for a named note. A4 = 440 Hz."""
    idx = NOTES.index(note_name)
    midi = 12 * (octave + 1) + idx
    return 440.0 * (2 ** ((midi - 69) / 12))


def scale_freqs(root, scale_name, octave=4, n_octaves=2):
    """Get all frequencies in a scale across octaves."""
    root_idx = NOTES.index(root)
    intervals = SCALES[scale_name]
    freqs = []
    for oct in range(octave, octave + n_octaves):
        for interval in intervals:
            midi = 12 * (oct + 1) + root_idx + interval
            freqs.append(440.0 * (2 ** ((midi - 69) / 12)))
    # Add the top note
    midi = 12 * (octave + n_octaves + 1) + root_idx
    freqs.append(440.0 * (2 ** ((midi - 69) / 12)))
    return freqs


# ============================================================
# DSP
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


def lowpass(signal, cutoff, order=4):
    cutoff = min(cutoff, SR / 2 - 1)
    sos = butter(order, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, signal)


def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return signal
    return signal / peak


def delay(signal, delay_sec=0.3, feedback=0.3, mix=0.25):
    """Simple feedback delay."""
    delay_samples = int(SR * delay_sec)
    out = signal.copy()
    for i in range(delay_samples, len(out)):
        out[i] += out[i - delay_samples] * feedback
    return signal * (1 - mix) + out * mix


def reverb(signal, decay=0.4, n_taps=6):
    """Cheap comb-filter reverb."""
    out = signal.copy()
    rng = np.random.default_rng(42)
    delays = rng.integers(int(SR * 0.02), int(SR * 0.08), size=n_taps)
    for i, d in enumerate(delays):
        amp = decay ** (i + 1)
        shifted = np.zeros(len(signal))
        shifted[d:] = signal[:-d] * amp
        out += shifted
    return out


# ============================================================
# Instrument voices (built from technique generators)
# ============================================================

def voice_ghost_hat(freq, dur, rng):
    """Ghost note hi-hat pattern."""
    notes = int(dur * 4)  # 16th-note-ish density
    slot = dur / max(notes, 1)
    out = np.zeros(int(SR * dur))
    for i in range(notes):
        start = int(SR * slot * i)
        seg_dur = slot * 0.4
        n_samp = int(SR * seg_dur)
        noise = rng.normal(0, 1, n_samp)
        env = np.exp(-25 * np.linspace(0, 1, n_samp))
        seg = lowpass(noise * env, freq * 3)
        # Ghost: every other hit is quiet
        if i % 2 == 1:
            seg *= rng.uniform(0.08, 0.15)
        else:
            seg *= 0.6
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] += seg
    return out


def voice_kick(freq, dur, rng):
    """Kick drum: sine with pitch drop."""
    tt = t(dur)
    # Pitch drops from ~150 Hz to ~50 Hz
    pitch = 150 * np.exp(-15 * tt / dur) + 50
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    sig = np.sin(phase)
    env = np.exp(-6 * tt / dur)
    return sig * env


def voice_snare(freq, dur, rng):
    """Snare hit: noise + tone."""
    tt = t(dur)
    tone = np.sin(2 * np.pi * 180 * tt)
    noise = rng.normal(0, 1, len(tt))
    noise = lowpass(noise, 8000)
    env = np.exp(-12 * tt / dur)
    return (tone * 0.4 + noise * 0.6) * env


def voice_roll_fill(freq, dur, rng):
    """Snare roll fill."""
    rate = rng.uniform(18, 28)
    n_hits = int(rate * dur)
    out = np.zeros(int(SR * dur))
    hit_dur = 0.025
    for i in range(n_hits):
        pos = i / rate
        start = int(SR * pos)
        n_samp = int(SR * hit_dur)
        if start + n_samp > len(out):
            break
        hit = rng.normal(0, 1, n_samp) * 0.5 + np.sin(2 * np.pi * 180 * np.linspace(0, hit_dur, n_samp)) * 0.5
        hit *= np.exp(-30 * np.linspace(0, 1, n_samp))
        # Crescendo through the fill
        hit *= 0.3 + 0.7 * (i / max(n_hits - 1, 1))
        out[start:start + n_samp] += hit
    return out


def voice_pizz_bass(freq, dur, rng):
    """Plucked bass note."""
    tt = t(dur)
    sig = np.zeros(len(tt))
    for k in range(1, 5):
        sig += (1.0 / k) * np.sin(2 * np.pi * freq * k * tt)
    decay = rng.uniform(5, 9)
    env = np.exp(-decay * tt / dur)
    return sig * env


def voice_arpeggio(freqs, dur, rng):
    """Broken chord across given frequencies."""
    n = len(freqs)
    note_dur = dur / n
    out = np.zeros(int(SR * dur))
    for i, f in enumerate(freqs):
        start = int(SR * note_dur * i)
        seg = sine(f, note_dur)
        decay = np.exp(-rng.uniform(2, 4) * np.linspace(0, 1, len(seg)))
        seg *= decay * 0.7
        fade = int(SR * 0.005)
        if fade > 0 and fade < len(seg):
            seg[:fade] *= np.linspace(0, 1, fade)
        end = start + len(seg)
        if end > len(out):
            seg = seg[:len(out) - start]
            end = len(out)
        out[start:end] = seg
    return out


def voice_pad(freq, dur, rng):
    """Sustained pad with slow attack, vibrato."""
    tt = t(dur)
    # Vibrato
    vib_rate = rng.uniform(4, 6)
    vib_depth = rng.uniform(2, 5)
    inst_freq = freq + vib_depth * np.sin(2 * np.pi * vib_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase) * 0.5
    # Add a detuned layer
    sig += np.sin(phase * 1.003) * 0.3
    sig += np.sin(phase * 0.998) * 0.3
    # Slow attack and release
    attack = int(SR * min(0.3, dur * 0.15))
    release = int(SR * min(0.4, dur * 0.2))
    env = np.ones(len(tt))
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return sig * env


def voice_lead(freq, dur, rng):
    """Distorted lead tone."""
    sig = sawtooth(freq, dur)
    sig = np.tanh(sig * 3)
    sig = lowpass(sig, freq * 5)
    # Slight vibrato
    tt = t(dur)
    mod = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * tt)
    inst_freq = freq * mod
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.tanh(sawtooth(freq, dur) * 3)
    sig = lowpass(sig, freq * 5)
    env = np.ones(len(tt))
    attack = int(SR * 0.01)
    release = int(SR * min(0.05, dur * 0.1))
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return sig * env


def voice_flutter_texture(freq, dur, rng):
    """Flutter tongue texture."""
    tt = t(dur)
    flutter_rate = rng.uniform(22, 30)
    inst_freq = freq + rng.uniform(2, 4) * np.sin(2 * np.pi * flutter_rate * tt)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase)
    amp_mod = 0.6 + 0.4 * np.sin(2 * np.pi * flutter_rate * tt)
    env = np.ones(len(tt))
    attack = int(SR * 0.1)
    release = int(SR * 0.2)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return sig * amp_mod * env


# ============================================================
# Mixer
# ============================================================

class Track:
    """A named audio track with volume and pan."""
    def __init__(self, name, n_samples, volume=1.0, pan=0.0):
        self.name = name
        self.audio = np.zeros(n_samples)
        self.volume = volume
        self.pan = pan  # -1 = hard left, 0 = center, 1 = hard right

    def place(self, signal, time_sec):
        start = int(SR * time_sec)
        end = start + len(signal)
        if end > len(self.audio):
            signal = signal[:len(self.audio) - start]
            end = len(self.audio)
        if start < len(self.audio):
            self.audio[start:end] += signal

    def render_stereo(self):
        """Return (left, right) arrays."""
        left_amp = np.sqrt(0.5 * (1 - self.pan))
        right_amp = np.sqrt(0.5 * (1 + self.pan))
        scaled = self.audio * self.volume
        return scaled * left_amp, scaled * right_amp


def mix_tracks(tracks):
    """Mix tracks to stereo, normalize."""
    n = max(len(tr.audio) for tr in tracks)
    left = np.zeros(n)
    right = np.zeros(n)
    for tr in tracks:
        l, r = tr.render_stereo()
        left[:len(l)] += l
        right[:len(r)] += r
    left = normalize(left) * 0.85
    right = normalize(right) * 0.85
    return np.column_stack([left, right])


# ============================================================
# Song builder
# ============================================================

def build_song(root='A', scale_name='minor_pent', bpm=85, bars=16, seed=42):
    rng = np.random.default_rng(seed)

    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4
    total_dur = bar_dur * bars
    n_samples = int(SR * total_dur)

    freqs = scale_freqs(root, scale_name, octave=3, n_octaves=3)
    bass_freqs = scale_freqs(root, scale_name, octave=2, n_octaves=1)
    chord_root = note_freq(root, 4)

    # Chord progression (scale degrees)
    if scale_name in ('minor', 'minor_pent', 'blues', 'harmonic_minor', 'dorian'):
        prog = [0, 3, 5, 3]  # i - iv - v - iv ish
    else:
        prog = [0, 3, 4, 0]  # I - IV - V - I
    scale_intervals = SCALES[scale_name]

    # --- Tracks ---
    tr_kick   = Track("kick",    n_samples, volume=0.9, pan=0.0)
    tr_hat    = Track("hat",     n_samples, volume=0.45, pan=0.3)
    tr_snare  = Track("snare",   n_samples, volume=0.7, pan=-0.1)
    tr_bass   = Track("bass",    n_samples, volume=0.75, pan=0.0)
    tr_arp    = Track("arp",     n_samples, volume=0.35, pan=-0.4)
    tr_pad    = Track("pad",     n_samples, volume=0.3, pan=0.0)
    tr_lead   = Track("lead",    n_samples, volume=0.4, pan=0.2)
    tr_perc   = Track("perc",    n_samples, volume=0.5, pan=0.0)

    for bar in range(bars):
        bar_time = bar * bar_dur
        chord_deg = prog[bar % len(prog)]

        # Determine which sections are active (arrangement)
        section = bar // 4  # 4-bar sections
        has_drums = bar >= 2
        has_bass = bar >= 2
        has_arp = 1 <= section
        has_pad = section >= 1
        has_lead = section >= 2 and bar % 2 == 0
        has_fill = has_drums and (bar % 4 == 3)  # fill on last bar of each section

        # -- Kick: beats 1 and 3 --
        if has_drums:
            for beat in [0, 2]:
                tr_kick.place(voice_kick(50, beat_dur * 0.6, rng), bar_time + beat * beat_dur)
            # Add an extra kick on beat 4 sometimes
            if rng.random() < 0.3:
                tr_kick.place(voice_kick(50, beat_dur * 0.4, rng) * 0.7,
                              bar_time + 3.5 * beat_dur)

        # -- Hi-hat: ghost note pattern --
        if has_drums:
            tr_hat.place(voice_ghost_hat(8000, bar_dur, rng), bar_time)

        # -- Snare: beats 2 and 4 --
        if has_drums:
            for beat in [1, 3]:
                tr_snare.place(voice_snare(200, beat_dur * 0.4, rng), bar_time + beat * beat_dur)

        # -- Fill on section boundaries --
        if has_fill:
            fill_start = bar_time + 3 * beat_dur
            tr_perc.place(voice_roll_fill(200, beat_dur, rng), fill_start)

        # -- Bass: root note of chord, plucked --
        if has_bass:
            bass_note = bass_freqs[min(chord_deg, len(bass_freqs) - 1)]
            for beat in range(4):
                if beat == 0 or rng.random() < 0.4:
                    tr_bass.place(voice_pizz_bass(bass_note, beat_dur * 0.8, rng),
                                  bar_time + beat * beat_dur)

        # -- Arpeggio --
        if has_arp:
            root_idx = chord_deg
            arp_freqs = []
            for degree_offset in [0, 2, 4, 2]:
                idx = (root_idx + degree_offset) % len(freqs)
                arp_freqs.append(freqs[idx])
            for beat in range(2):
                tr_arp.place(voice_arpeggio(arp_freqs, beat_dur * 2, rng),
                             bar_time + beat * 2 * beat_dur)

        # -- Pad: sustained chord --
        if has_pad:
            pad_freq = freqs[min(chord_deg, len(freqs) - 1)]
            tr_pad.place(voice_pad(pad_freq, bar_dur, rng), bar_time)

        # -- Lead melody --
        if has_lead:
            n_notes = rng.integers(3, 7)
            positions = sorted(rng.choice(8, size=n_notes, replace=False))
            for pos in positions:
                note_time = bar_time + pos * (beat_dur / 2)
                note_dur = beat_dur * rng.uniform(0.4, 1.2)
                note_freq_val = rng.choice(freqs[4:12])
                if rng.random() < 0.3:
                    tr_lead.place(voice_flutter_texture(note_freq_val, note_dur, rng), note_time)
                else:
                    tr_lead.place(voice_lead(note_freq_val, note_dur, rng), note_time)

    # --- Mix ---
    tracks = [tr_kick, tr_hat, tr_snare, tr_bass, tr_arp, tr_pad, tr_lead, tr_perc]

    # Apply effects to individual tracks
    tr_snare.audio = reverb(tr_snare.audio, decay=0.3, n_taps=4)
    tr_arp.audio = delay(tr_arp.audio, delay_sec=beat_dur * 0.75, feedback=0.3, mix=0.2)
    tr_lead.audio = delay(tr_lead.audio, delay_sec=beat_dur * 0.5, feedback=0.25, mix=0.15)
    tr_pad.audio = reverb(tr_pad.audio, decay=0.5, n_taps=8)

    stereo = mix_tracks(tracks)
    return stereo


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Synth song builder")
    parser.add_argument("--bpm", type=int, default=85)
    parser.add_argument("--key", type=str, default="A")
    parser.add_argument("--scale", type=str, default="minor_pent",
                        choices=list(SCALES.keys()))
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="synth_song.wav")
    args = parser.parse_args()

    print(f"Building: {args.key} {args.scale}, {args.bpm} BPM, {args.bars} bars")
    stereo = build_song(
        root=args.key,
        scale_name=args.scale,
        bpm=args.bpm,
        bars=args.bars,
        seed=args.seed,
    )

    pcm = (stereo * 32767).astype(np.int16)
    write(args.out, SR, pcm)

    duration = len(stereo) / SR
    print(f"Saved: {args.out} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
