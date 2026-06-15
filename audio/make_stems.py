import argparse
import os
import warnings
from glob import glob

import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model
from tqdm import tqdm

# demucs.save_audio -> torchaudio.save passes params (bits_per_sample, encoding, ...)
# that TorchCodec's AudioEncoder ignores; the write still succeeds. Silence the
# whole benign family.
warnings.filterwarnings("ignore",
                        message=r"The '.*' parameter is not .* supported by TorchCodec AudioEncoder")


def is_junk(path):
    return os.path.basename(path).startswith("._") or "__MACOSX" in path


def main(data_dir, model_name="htdemucs", device=None):
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")

    model = get_model(model_name)
    model.to(device)
    model.eval()

    base = data_dir.rstrip("/")
    wav_files = sorted(f for f in glob(os.path.join(base, "**/*.wav"), recursive=True) if not is_junk(f))
    mp3_files = sorted(f for f in glob(os.path.join(base, "**/*.mp3"), recursive=True) if not is_junk(f))
    flac_files = sorted(f for f in glob(os.path.join(base, "**/*.flac"), recursive=True) if not is_junk(f))
    files = wav_files + mp3_files + flac_files

    print(f"Separating {len(files)} files with '{model_name}' on {device} "
          f"-> stems {model.sources}")

    for path in tqdm(files, desc="Separating stems"):
        wav = AudioFile(path).read(streams=0, samplerate=model.samplerate,
                                   channels=model.audio_channels)
        # Demucs expects per-track normalized input, then rescales the outputs back.
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device, progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        rel = os.path.relpath(path, base)  # same path layout inside each data-{stem}/
        for name, source in zip(model.sources, sources):
            out_path = os.path.join(f"{base}-{name}", rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_audio(source.cpu(), out_path, samplerate=model.samplerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clone data/ into data-{stem}/ trees via Demucs source separation.")
    parser.add_argument("--data", dest="data_dir", required=True)
    parser.add_argument("--model", default="htdemucs", dest="model_name",
                        help="Demucs model. htdemucs = 4 stems (drums/bass/other/vocals); "
                             "htdemucs_6s adds guitar/piano (experimental quality).")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset).")
    args = parser.parse_args()
    main(args.data_dir, args.model_name, args.device)
