#!/bin/bash
# Full pipeline for every (dataset x stem x normalization) combo.
#
#   datasets : the 9 per-dataset wrappers (ds002725 excluded -- 024 duplicate)
#   stems    : raw mix + the 4 demucs stems
#   norms    : none / --per-chunk / --per-band / both
#
# = 9 x 5 x 4 = 180 runs of `<dataset>.py --all`. Combos whose data dir
# doesn't exist yet (e.g. stems not separated) are skipped, not failed.
# No `set -e`: one bad run must not abort the overnight sweep -- failures are
# logged and the loop keeps going.
#
# > Thanks Claude

cd "$(dirname "$0")"

PY="conda run -n musicmri python"

# Floor the min_cluster_size sweep so silhouette can't pick the degenerate
# mcs=5 cut (which produced 100+ micro-clusters on near-1D embeddings).
MCS="--mcs-min 20"

# Energy gate (dBFS) applied to stems only. An absent instrument's demucs
# residual is near-silent; without the gate per-band normalization amplifies it
# into fake clusters (e.g. ds002722's piano -> silent drum stem). The raw mix is
# never silent, so it runs ungated.
STEM_GATE="--min-rms -50"

DATASETS=(ds002721 ds002722 ds002724 ds003720 ds003774 NMED-E NMED-H NMED-M NMED-T)
STEMS=(raw bass drums other vocals)               # "raw" = the mix (no --stem)
NORMS=("" "--per-chunk" "--per-band" "--per-chunk --per-band")

LOG="sweep_$(date +%Y%m%d_%H%M%S).log"
declare -a FAILED

echo "Sweep started $(date) -> $LOG"

for ds in "${DATASETS[@]}"; do
    for stem in "${STEMS[@]}"; do
        if [ "$stem" = raw ]; then
            data_dir="data/data-raw/$ds"
            stem_flag=""
            gate=""
        else
            data_dir="data/data-$stem/$ds"
            stem_flag="--stem $stem"
            gate="$STEM_GATE"
        fi

        if [ ! -d "$data_dir" ]; then
            echo "skip: $ds stem=$stem (no $data_dir)" | tee -a "$LOG"
            continue
        fi

        for norm in "${NORMS[@]}"; do
            label="$ds stem=$stem norm=${norm:-none}"
            echo "=== $label ===" | tee -a "$LOG"
            if $PY "$ds.py" --all $stem_flag $norm $MCS $gate >>"$LOG" 2>&1; then
                echo "ok:   $label" | tee -a "$LOG"
            else
                echo "FAIL: $label" | tee -a "$LOG"
                FAILED+=("$label")
            fi
        done
    done
done

echo "Done $(date). ${#FAILED[@]} failure(s)." | tee -a "$LOG"
printf '  %s\n' "${FAILED[@]}" | tee -a "$LOG"
