# Video downloading

[Dataset guide](../README.md)

Original videos are not hosted with the dataset. Download `mugen.csv` from the
Hugging Face annotation link in the [dataset guide](../README.md#annotations), then run from `MUGEN/dataset/`
with Python 3.10+, FFmpeg and yt-dlp installed.

```bash
python dataset_downloading/csv_to_urls.py --input_file mugen.csv --output_file mugen_urls.txt
yt-dlp --ignore-config --no-playlist -N 10 -f "bestvideo+bestaudio/best" \
  --format-sort-force -S "res,fps" --merge-output-format mkv --remux-video mkv \
  --write-info-json -a mugen_urls.txt -o "videos/%(id)s.%(ext)s"
```

This selects resolution first and retains download metadata for later auditing.
See the [official yt-dlp format selection documentation](https://github.com/yt-dlp/yt-dlp#format-selection).
Source resolution is preserved during transcoding. Available formats may change.

Use the same CSV for both downloading and clip extraction. For the HQ selection,
use `mugen-hq.csv`. The source reader accepts either
`videos/VIDEO_ID.mkv` or `videos/VIDEO_ID/VIDEO_ID.mkv` (also MP4/WebM).
If multiple candidates exist, select one explicitly by placing it in a separate
input directory; the pipeline does not silently choose between different resolutions.
Missing downloads fail processing and must be resolved or explicitly excluded in a
separate local CSV. Keep the downloaded release CSV unchanged.
