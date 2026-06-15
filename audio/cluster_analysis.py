import argparse
import json
import re
from collections import Counter, defaultdict

FILE_REGEX = re.compile(r"data/data-[^/]*/(ds\d+|NMED-[A-Z])/(.+)$")


def dataset_counts(clusters):
    ds_counts = defaultdict(Counter)
    
    for cluster, data in clusters.items():
        cluster = int(cluster)
        for item in data:
            match = FILE_REGEX.search(item["file"])
            if match:
                ds_counts[cluster][match.group(1)] += 1
    
    return ds_counts


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clusters_file", help="Path to a model's clusters.json.")
    args = p.parse_args()

    with open(args.clusters_file) as f:
        clusters = json.load(f)

    ds_counts = dataset_counts(clusters)
    for cluster, counts in sorted(ds_counts.items(), key=lambda x: int(x[0])):
        print(f"Cluster {cluster}:")
        for ds_num, count in counts.items():
            print(f"  Dataset {ds_num}: {count} stimuli")


if __name__ == "__main__":
    main()
