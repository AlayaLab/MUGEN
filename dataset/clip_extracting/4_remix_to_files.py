"""Mux MUGEN clips at 30 FPS with correct presentation timestamps for HEVC B frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import shutil
import tempfile
from utils.manifest import read_clips


def run(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True)


def process_one_video(args, metadata):
    vid = metadata.stem
    clips = read_clips(metadata)
    output = Path(args.output_dir) / vid
    output.mkdir(parents=True, exist_ok=True)
    audio = Path(args.input_astream_dir or ".") / (vid + ".flac")
    if not args.ignore_audio and not audio.is_file():
        raise FileNotFoundError(audio)
    with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
        tmp = Path(tmp)
        for start, end in clips:
            fid = f"{vid}_{start:07d}_{end:07d}"
            video = Path(args.input_vstream_dir) / vid / (fid + ".hevc")
            if not video.is_file():
                raise FileNotFoundError(video)
            mkv = tmp / "video.mkv"
            aac = tmp / "audio.m4a"
            target = output / (fid + ".mp4")
            partial = output / (fid + ".partial.mp4")
            try:
                # Raw HEVC has no container timestamps. mkvmerge uses picture order
                # to reconstruct PTS for B frames; assigning PTS=DTS via AVI is wrong.
                subprocess.run([
                    "mkvmerge", "-q", "-o", str(mkv), "--timestamp-scale", "1",
                    "--default-duration", "0:30fps", str(video),
                ], check=True)
                inputs = ["-i", mkv]
                maps = ["-map", "0:v:0"]
                if not args.ignore_audio:
                    run("-i", audio, "-ss", f"{start / 30:.6f}", "-to", f"{end / 30:.6f}",
                        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k", aac)
                    inputs += ["-i", aac]
                    maps += ["-map", "1:a:0"]
                run(*inputs, *maps, "-c", "copy", "-vtag", "hvc1", "-movflags", "+faststart",
                    "-video_track_timescale", "30000", partial)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_clip_dir", required=True)
    parser.add_argument("--input_astream_dir")
    parser.add_argument("--input_vstream_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temp_dir")
    parser.add_argument("--ignore_audio", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if shutil.which("mkvmerge") is None:
        parser.error("mkvmerge is required; install MKVToolNix (e.g. apt install mkvtoolnix)")
    if args.num_workers < 1 or (not args.ignore_audio and not args.input_astream_dir):
        parser.error("Require positive workers and --input_astream_dir unless --ignore_audio")
    metadata = sorted(Path(args.input_clip_dir).glob("*.txt"))
    if not metadata:
        parser.error("No clip metadata found")
    with ThreadPoolExecutor(args.num_workers) as pool:
        list(pool.map(lambda p: process_one_video(args, p), metadata))
