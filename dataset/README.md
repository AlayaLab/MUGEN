# MUGEN dataset

[Project overview](../README.md) · [Wan360 model](../model/README.md)

Download and reconstruct MUGEN panoramic video clips using the Sekai release workflow. MUGEN is one dataset; all sources use their highest available resolution.

## Dataset overview

| Property | Value |
| --- | --- |
| Released clip entries | 77,023 |
| Source videos | 6,230 |
| HQ subset | 17,655 clips from the same dataset |
| Clip duration | 60 seconds / 1,800 frames at 30 FPS |
| Total duration | Approximately 1,283.7 hours |
| Video layout | 2:1 equirectangular panorama |

The release list includes currently generated clips. Original videos are **not distributed**: download the sources from YouTube and reconstruct clips from the published CSV.

## Annotations

| Resource | Download / status |
| --- | --- |
| Hugging Face repository | [AlayaLab/MUGEN](https://huggingface.co/datasets/AlayaLab/MUGEN) |
| Full CSV: captions and categories | [mugen.csv](https://huggingface.co/datasets/AlayaLab/MUGEN/resolve/main/train/mugen.csv) |
| HQ subset CSV: captions and categories | [mugen-hq.csv](https://huggingface.co/datasets/AlayaLab/MUGEN/resolve/main/train/mugen-hq.csv) |
| Camera trajectories | [pose.tar](https://huggingface.co/datasets/AlayaLab/MUGEN/resolve/main/annotations/pose.tar) |
| Depth | `TO BE RELEASE` |

Category annotations are already included in both CSVs. Camera trajectories are available as `pose.tar`, containing all 77,023 CSV-referenced NPZ files. Depth is pending release. The CSV columns are `videoFile`, `cameraFile`, `caption`, `location`, `scene`, `crowdDensity`, `weather` and `timeOfDay`.

## Reconstruct the dataset

Run dataset commands from **`MUGEN/dataset/`**. The processing pipeline requires FFmpeg, MKVToolNix and an NVIDIA GPU with suitable decoding and HEVC encoding support; dependency details are in the extraction guide.

1. Download `mugen.csv` using the link above. For the HQ subset, use `mugen-hq.csv` throughout the workflow.
2. Follow the [source download guide](dataset_downloading/README.md) to create YouTube URLs and download each source at the highest available resolution.
3. Follow the [clip extraction guide](clip_extracting/README.md) to create clip intervals, separate audio, transcode video and mux the final clips.
4. Download `pose.tar` from the link above and extract it with `tar -xf pose.tar`. Match `pose/<cameraFile>` to each CSV row. Depth is pending release.

Reconstructed clips are written to `work/files/VIDEO_ID/FID.mp4`. Clip IDs encode frame intervals on the processed 30 FPS timeline. Source availability and downloaded versions may change; timing and resolution details are documented in the extraction guide.

## Detailed guides

| Directory | Documentation |
| --- | --- |
| `dataset_downloading/` | [Source download commands and file layout](dataset_downloading/README.md) |
| `clip_extracting/` | [Dependencies, processing commands, timing and resolution](clip_extracting/README.md) |

## License

MUGEN annotations, including captions, category annotations, camera trajectories and depth when released, are licensed under [CC BY 4.0](LICENSE-ANNOTATIONS). Attribute MUGEN, link to the license and indicate any changes when sharing licensed material. This license covers only rights held by the MUGEN contributors in the annotations and database, and does not grant rights over source videos or reconstructed video clips.

Processing code is adapted from Sekai and retains its original non-commercial research [LICENSE](LICENSE). The code and annotations have separate licenses.
