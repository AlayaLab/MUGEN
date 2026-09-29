import argparse
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".csv", "_urls.txt")

    vids = set()

    with open(args.input_file, "r") as f:
        csv_reader = csv.reader(f)
        for idx, line in enumerate(csv_reader):
            if idx == 0:
                continue

            vids.add(os.path.splitext(line[0])[0].rsplit("_", 2)[0])

    with open(args.output_file, "x") as f:
        [f.write(f"https://www.youtube.com/watch?v={vid}\n") for vid in vids]
