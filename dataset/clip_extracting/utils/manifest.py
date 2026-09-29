"""Shared MUGEN identifiers and local input discovery (standard library only)."""
import re
from pathlib import Path


def read_fids(filename):
    groups = {}
    seen = set()
    for number, line in enumerate(Path(filename).read_text().splitlines(), 1):
        fid = line.strip().removesuffix(".mp4")
        if not fid:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]{11})_(\d{7})_(\d{7})", fid)
        if not match:
            raise ValueError(f"Invalid fid at line {number}: {fid!r}")
        vid, start, end = match.groups()
        start, end = int(start), int(end)
        if start >= end or fid in seen:
            raise ValueError(f"Duplicate or invalid frame range at line {number}: {fid}")
        seen.add(fid)
        groups.setdefault(vid, []).append((start, end))
    for vid, clips in groups.items():
        clips.sort()
        if any(a[1] > b[0] for a, b in zip(clips, clips[1:])):
            raise ValueError(f"Overlapping clips for {vid}")
    if not groups:
        raise ValueError("Empty fid manifest")
    return dict(sorted(groups.items()))


def read_clips(filename):
    clips = [tuple(map(int, line.split())) for line in Path(filename).read_text().splitlines() if line.strip()]
    clips.sort()
    if not clips or any(len(c) != 2 or c[0] < 0 or c[0] >= c[1] for c in clips):
        raise ValueError(f"Invalid clips: {filename}")
    if any(a[1] > b[0] for a, b in zip(clips, clips[1:])):
        raise ValueError(f"Overlapping clips: {filename}")
    return clips


def find_video(directory, vid):
    candidates = [p for base in (Path(directory), Path(directory) / vid)
                  for ext in (".mkv", ".mp4", ".webm")
                  if (p := base / (vid + ext)).is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one source video for {vid}, found {candidates}")
    return candidates[0]
