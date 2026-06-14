import json
import re
from collections import defaultdict, Counter

FILE_REGEX = re.compile(r"^data/ds(\d+)/stimuli/(.+)$")

CLUSTERS_FILE = "./models/clusters.json"

with open(CLUSTERS_FILE, "r") as f:
    clusters = json.load(f)
    ds_counts = defaultdict(Counter)

    for cluster, data in clusters.items():
        cluster = int(cluster)
        for item in data:
            file = item["file"]
            match = FILE_REGEX.match(file)
            if match:
                ds_num = match.group(1)
                stimulus = match.group(2)
                ds_counts[cluster][ds_num] += 1

for cluster, counts in sorted(ds_counts.items(), key=lambda x: int(x[0])):
    print(f"Cluster {cluster}:")
    for ds_num, count in counts.items():
        print(f"  Dataset {ds_num}: {count} stimuli")