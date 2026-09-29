"""Extract stereo FLAC, matching the MUGEN near-30 FPS tempo correction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from utils.manifest import find_video


def extract_astream(video, output):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=r_frame_rate", "-of", "json", str(video)]))
    fps = float(Fraction(info["streams"][0]["r_frame_rate"]))
    filters = ["-af", f"atempo={30 / fps:.6f}"] if 0.01 < abs(fps - 30) < 0.5 else []
    if fps < 29.5:
        raise ValueError(f"{video}: source FPS {fps} requires a separately verified audio timing policy")
    temp = output.with_suffix(".partial.flac")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0",
                        *filters, "-c:a", "flac", "-ar", "48000", "-ac", "2", "-sample_fmt",
                        "s16", str(temp)], check=True)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--input_clip_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num_workers must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(find_video(args.input_dir, p.stem), output / (p.stem + ".flac"))
            for p in sorted(Path(args.input_clip_dir).glob("*.txt"))]
    if not jobs:
        parser.error("No clip metadata found")
    with ThreadPoolExecutor(args.num_workers) as pool:
        list(pool.map(lambda job: extract_astream(*job), jobs))
