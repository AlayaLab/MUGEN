import argparse
import csv
import os
from collections import defaultdict
from operator import itemgetter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_file.replace(".csv", "")

    os.makedirs(args.output_dir, exist_ok=False)

    vid2clips = defaultdict(list)

    with open(args.input_file, "r") as f:
        csv_reader = csv.reader(f)
        for idx, line in enumerate(csv_reader):
            if idx == 0:
                continue
            vid, s_frame, e_frame = os.path.splitext(line[0])[0].rsplit("_", 2)
            vid2clips[vid].append((s_frame, e_frame))

    for vid, clips in vid2clips.items():
        clips = sorted(clips, key=itemgetter(0))

        with open(os.path.join(args.output_dir, vid + ".txt"), "x") as f:
            [f.write(" ".join(clip) + "\n") for clip in clips]
