#!/bin/bash
set -euo pipefail

for data_type in "raw" "bass" "drums" "other" "vocals"; do
    echo "Clustering on ${data_type} data..."
    python cluster.py train --data_dir data/data-${data_type} --model_dir models/${data_type}
    echo "    data->${data_type} -> models/${data_type} (base)..."
    python cluster.py train --data_dir data/data-${data_type} --model_dir models/${data_type}-chunk --per-chunk
    echo "    data->${data_type} -> models/${data_type} (per-band)..."
    python cluster.py train --data_dir data/data-${data_type} --model_dir models/${data_type}-band --per-band
    echo "    data->${data_type} -> models/${data_type} (per-chunk and per-band)..."
    python cluster.py train --data_dir data/data-${data_type} --model_dir models/${data_type}-chunk-band --per-chunk --per-band
done

echo "Done!"