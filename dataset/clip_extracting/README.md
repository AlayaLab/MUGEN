# Clip extraction

[Dataset guide](../README.md)

Run all commands from `MUGEN/dataset/`. CPU stages need Python 3.10+,
FFmpeg/ffprobe and MKVToolNix (`mkvmerge`). On Ubuntu:

```bash
sudo apt install ffmpeg mkvtoolnix
python -m pip install -r clip_extracting/requirements.txt
```

The GPU runtime was exercised with Python 3.12, PyTorch 2.6.0+cu124,
PyNvVideoCodec 2.0.2, CV-CUDA 0.15.0, PyCUDA 2025.1.2 and an RTX 4090
with NVIDIA driver 550.163.01. `requirements.txt` records the tested direct
runtime dependencies; a fresh environment installation has not been verified.
A compatible CUDA toolkit is also needed when building PyCUDA from source.
The GPU must support decoding the source codec and encoding HEVC at its full width.

```bash
python clip_extracting/1_csv_to_clips.py --input_file mugen.csv --output_dir work/clips
python clip_extracting/2_split_audios.py --input_clip_dir work/clips --input_dir videos --output_dir work/astreams --num_workers 4
python clip_extracting/3_nvtranscoding.py --input_clip_dir work/clips --input_video_dir videos --output_dir work/vstreams --device_id 0
python clip_extracting/4_remix_to_files.py --input_clip_dir work/clips --input_astream_dir work/astreams --input_vstream_dir work/vstreams --output_dir work/files --num_workers 4
```

Download `mugen.csv` from the Hugging Face annotation link in the [dataset guide](../README.md#annotations).
Use `mugen-hq.csv` instead for the HQ selection. All MUGEN clips use the same pipeline. Outputs are `work/files/VIDEO_ID/FID.mp4`.
Use `--ignore_audio` in the final step for mute clips, omitting audio extraction.
`--temp_dir` controls intermediate storage. The clip metadata step requires a new output directory. Audio, video and mux
reruns regenerate outputs rather than
assuming an existing directory means successful completion. Final files are replaced
only after each corresponding command succeeds; failures return a nonzero exit status.
Do not run multiple processes against the same video/output paths.

## Timing and resolution

A fid has the form `VIDEO_ID_START_END`; the 11-character ID can itself contain
underscores. Start is inclusive and end exclusive, measured on the processed
30 FPS timeline. The sampler preserves all frames below 30 FPS and downsamples
higher frame rates using the original Sekai/MUGEN sampler.
For `0.01 < abs(source_fps - 30) < 0.5`, audio uses `atempo=30/source_fps`, matching
the local MUGEN script. Sources below 29.5 FPS fail explicitly: the legacy pipeline
kept their frames without a corresponding audio correction, so they require review.

Output width comes from the decoded source and height is width / 2, matching
the original MUGEN processing pipeline. Legacy `--width`/`--height` options only set
initial dimensions and are overridden per source. FPS must be 30 and batch size 1.
Unsupported pixel formats or insufficient encoder capacity fail explicitly.

The video stage produces raw HEVC. The mux stage uses `mkvmerge` to reconstruct
30 FPS presentation timestamps through a Matroska intermediate, then copies video
and AAC audio into MP4. This corrects the legacy AVI path, which incorrectly
assigned PTS=DTS for B-frame HEVC and caused timestamps to go backward after decoding.
Video is not re-encoded. See [mkvmerge's timing options](https://mkvtoolnix.download/doc/mkvmerge.html).
It writes local MP4 files; no remote upload is performed.
