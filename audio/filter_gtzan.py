import os
import pandas as pd

ds = "ds003720"
folders = [f"data/data-raw/{ds}", f"data/data-vocals/{ds}", f"data/data-guitar/{ds}", f"data/data-bass/{ds}", f"data/data-drums/{ds}", f"data/data-other/{ds}"]
core = f"data/data-raw/{ds}"

keep_tracks = []

for folder in os.listdir(core):
    if not "sub-" in folder:
        continue
    for file in os.listdir(os.path.join(core, folder, "func")):
        p = os.path.join(core, folder, "func", file)
        tsv = pd.read_csv(p, sep="\t", quotechar='"')
        for i, row in tsv.iterrows():
            genre = row["genre"]
            track_num = "{:05d}".format(int(row["track"]))
            genre = genre.replace("'", "")
            keep_tracks.append(f"{genre}.{track_num}.wav")

for folder in folders[:1]:
    folder = os.path.join(folder, "stimuli")
    for nested_folder in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, nested_folder)):
            if not file in keep_tracks:
                os.remove(os.path.join(folder, nested_folder, file))
