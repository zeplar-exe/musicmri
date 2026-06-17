"""Run clustering presets defined in an XML file (default run_presets.xml).

Each <preset> bundles one or more datasets/stems into a SINGLE pooled model.
Included files are staged into a temp dir, preserving the
data/data-<stem>/<dataset>/ layout so dataset provenance survives in the path
strings written to clusters.json (cluster_analysis.py parses those). The
cluster.py CLI then runs against the temp dir, which is removed once every
stage finishes. Presets with disabled="true" are skipped.

    python run_presets.py                  # run_presets.xml, all stages
    python run_presets.py my.xml --train   # only the train stage
    python run_presets.py --profile        # reuse models, re-derive + re-plot

All stages need the staged audio present, so they run together in one
invocation -- re-running a single stage later would stage a fresh temp dir.

> Thanks Claude
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from glob import glob

AUDIO_EXTS = (".wav", ".mp3", ".flac")
STAGES = ["train", "profile", "exemplar"]

# cluster.py train CLI defaults, mirrored here so the saved preset.xml records the
# values actually used even when a preset leaves them unset. Keep in sync with
# cluster.py's train argparse. A None default (min_samples ties to min_cluster_size;
# min_rms = no energy gate) is written as an empty element, which parses back to None.
HDBSCAN_DEFAULTS = {"min_samples": None, "mcs_min": "5", "mcs_max": "50", "mcs_step": "5"}
MIN_RMS_DEFAULT = None


def _run(cmd, outfile=None):
    """Echo a cluster.py command and shell out to it,
    optionally capturing stdout to outfile. Raises on a non-zero exit."""
    shown = " ".join(cmd) + (f" > {outfile}" if outfile else "")
    print(f"\n+ {shown}")
    if outfile:
        with open(outfile, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh)
    else:
        subprocess.run(cmd, check=True)


def _load_presets(path, _seen=None):
    """Flat list of <preset> elements from a presets file, resolving any
    <import href="..."/> recursively. An href is relative to the file that
    declares it, so an index can pull in per-dataset files (e.g. sweep/ds*.xml).
    Document order is preserved and import cycles are rejected."""
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise RuntimeError(f"import cycle through {path}")
    _seen.add(path)

    base = os.path.dirname(path)
    presets = []
    for el in ET.parse(path).getroot():
        if el.tag == "import":
            href = el.get("href")
            if not href:
                raise RuntimeError(f"<import> without href in {path}")
            child = os.path.join(base, href)
            if not os.path.exists(child):
                raise FileNotFoundError(f"imported preset file not found: {child} (from {path})")
            presets.extend(_load_presets(child, _seen))
        elif el.tag == "preset":
            presets.append(el)
    return presets


def _stem_dir(stem):
    return "data-raw" if stem == "raw" else f"data-{stem}"


def _parse_preset(el):
    """Pull (name, prefix, includes, methods, hdbscan dict, min_rms) out of a <preset>."""
    prefix = el.get("prefix") or el.get("name")
    name = el.get("name") or prefix
    includes = [(d.get("stem", "raw"), (d.text or "").strip())
                for d in el.findall("include/dataset")]
    methods = [(m.text or "").strip() for m in el.findall("normalize/method")]

    hp = el.find("hdbscan")

    def hv(tag):
        e = hp.find(tag) if hp is not None else None
        return e.text.strip() if e is not None and e.text else None

    hdb = {
        "min_samples": hv("min_samples"),
        "mcs_min": hv("min_cluster_size_min"),
        "mcs_max": hv("min_cluster_size_max"),
        "mcs_step": hv("min_cluster_size_step"),
    }
    
    rms_el = el.find("min_rms")
    min_rms = rms_el.text.strip() if rms_el is not None and rms_el.text else None

    return name, prefix, includes, methods, hdb, min_rms


def _export_preset(el, hdb, min_rms):
    """Deep-copy the preset and backfill the hdbscan knobs + min_rms with
    cluster.py's defaults wherever the source left them unset, so the saved
    preset.xml is a complete record of the run. None (an unset min_samples /
    min_rms) is written as an empty element, which round-trips back to None."""
    el = copy.deepcopy(el)

    hp = el.find("hdbscan")
    if hp is None:
        hp = ET.SubElement(el, "hdbscan")
    for tag, key in (("min_samples", "min_samples"), ("min_cluster_size_min", "mcs_min"),
                     ("min_cluster_size_max", "mcs_max"), ("min_cluster_size_step", "mcs_step")):
        val = hdb[key] if hdb[key] is not None else HDBSCAN_DEFAULTS[key]
        sub = hp.find(tag)
        if sub is None:
            sub = ET.SubElement(hp, tag)
        sub.text = None if val is None else str(val)

    rms_el = el.find("min_rms")
    if rms_el is None:
        rms_el = ET.SubElement(el, "min_rms")
    rms_val = min_rms if min_rms is not None else MIN_RMS_DEFAULT
    rms_el.text = None if rms_val is None else str(rms_val)

    ET.indent(el, space="    ")  # readable provenance, not a one-liner
    return el


def _stage_files(includes, tmp):
    root = os.path.join(tmp, "data")
    n = 0
    
    for stem, dataset in includes:
        src = os.path.join("data", _stem_dir(stem), dataset)
        print(f"  staging from {src}...")
        for ext in AUDIO_EXTS:
            for f in glob(os.path.join(src, "**", f"*{ext}"), recursive=True):
                if os.path.basename(f).startswith("._") or "__MACOSX" in f:
                    continue
                dst = os.path.join(root, _stem_dir(stem), dataset, os.path.relpath(f, src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    
    return root, n


def _canonicalize_clusters(model_dir, tmp):
    """Rewrite clusters.json's file paths from the temp staging dir to the real paths."""
    path = os.path.join(model_dir, "clusters.json")
    with open(path) as f:
        clusters = json.load(f)
    for records in clusters.values():
        for r in records:
            r["file"] = os.path.relpath(r["file"], tmp)
    with open(path, "w") as f:
        json.dump(clusters, f, indent=2)  # preserve cluster.py's readable formatting


def run_preset(el, stages, py):
    name, prefix, includes, methods, hdb, min_rms = _parse_preset(el)
    print(f"\n##### preset {name} -> {prefix} #####")
    print("  includes:", ", ".join(f"{s}:{d}" for s, d in includes) or "(none)")

    norm_flags = (["--per-chunk"] if "chunk" in methods else []) \
        + (["--per-band"] if "band" in methods else [])
    hdb_flags = []
    
    for flag, key in (("--min-samples", "min_samples"), ("--mcs-min", "mcs_min"), ("--mcs-max", "mcs_max"), ("--mcs-step", "mcs_step")):
        if hdb[key] is not None:
            hdb_flags += [flag, hdb[key]]
    
    gate_flags = ["--min-rms", min_rms] if min_rms is not None else []

    model_dir = f"models/{prefix}"
    profiles_dir = f"{model_dir}/profiles"
    exemplars_dir = f"{model_dir}/exemplars"
    
    tmp = tempfile.mkdtemp(prefix=f"preset-{prefix}-")
    try:
        data_dir, n = _stage_files(includes, tmp)
        print(f"  staged {n} files -> {data_dir}")
        if n == 0:
            print("  nothing to stage; skipping preset.")
            return

        # save the preset that produced this model, with defaults filled in, for provenance.
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "preset.xml"), "w") as f:
            f.write(ET.tostring(_export_preset(el, hdb, min_rms), encoding="unicode"))

        if stages["train"]:
            _run([py, "cluster.py", "train", "--data", data_dir, "--model", model_dir,
                  *norm_flags, *hdb_flags, *gate_flags])
            _canonicalize_clusters(model_dir, tmp)
        if stages["profile"]:
            os.makedirs(profiles_dir, exist_ok=True)
            # profile now draws the heatmap + scatter from its single descriptor pass
            # (scatter reuses train's saved embedding), so there's no separate plot stage.
            _run([py, "cluster.py", "profile", "--model", model_dir, "--out-dir", profiles_dir],
                 outfile=f"{profiles_dir}/profile.txt")
        if stages["exemplar"]:
            if os.path.isdir(exemplars_dir):
                shutil.rmtree(exemplars_dir)
            _run([py, "cluster.py", "exemplars", "--model", model_dir, "--out", exemplars_dir])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nDone: {prefix}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("presets_file", nargs="?", default="run_presets.xml",
                   help="XML file of <preset> blocks (default run_presets.xml).")
    for s in STAGES:
        p.add_argument(f"--{s}", action="store_true", help=f"Run the {s} stage.")
    p.add_argument("--all", action="store_true", help="Run every stage (the default).")
    args = p.parse_args()

    stages = {s: getattr(args, s) or args.all for s in STAGES}
    if not any(stages.values()):  # no stage picked -> run the whole pipeline
        stages = {s: True for s in STAGES}

    py = sys.executable
    presets = _load_presets(args.presets_file)

    def is_true(el, attr):
        return (el.get(attr) or "").lower() == "true"

    solo = any(is_true(el, "enabled") for el in presets)
    if solo:
        print("solo mode: running only enabled=\"true\" presets")

    failed = []
    for el in presets:
        name = el.get("prefix") or el.get("name")
        if solo:
            if not is_true(el, "enabled"):
                print(f"skip (not enabled): {name}")
                continue
        elif is_true(el, "disabled"):
            print(f"skip (disabled): {name}")
            continue

        try:
            run_preset(el, stages, py)
        except Exception as e:
            print(f"\nFAIL: preset {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} preset(s) failed:")
        for name in failed:
            print(f"  {name}")


if __name__ == "__main__":
    main()
