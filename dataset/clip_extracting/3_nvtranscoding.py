# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise things may throw unexpected errors.
import gc
import logging
import os
import argparse
from pathlib import Path
from utils.manifest import find_video, read_clips

import pycuda.driver as cuda  # noqa: F401
import cvcuda

# import tqdm
import torch

from utils.nvvpf_utils import (
    VideoBatchDecoder,
    VideoMemoryEncoder,
)


def process_one_video(video_filename, vstream_filename_format, clips):
    files = []
    if len(clips) == 0:
        return files

    decoder.initialize(video_filename)

    if decoder.decoder.fps < 29.5:
        raise ValueError("Source FPS below 29.5 requires a separately verified timing policy")
    # Match nvtranscoding_8k.py: preserve source width and normalize ERP to 2:1.
    decoder.width = encoder.width = decoder.decoder.width
    decoder.height = encoder.height = decoder.decoder.width // 2
    if decoder.width % 4:
        raise ValueError("ERP source width must be divisible by 4 for NV12")

    clip_idx, s, e = 0, clips[0][0], clips[0][1]
    with cvcuda_stream, torch.cuda.stream(torch_stream):
        for frame_idx, frames in enumerate(decoder):
            if frame_idx < s:
                continue
            if frame_idx == s:
                encoder.initialize()
            encoder(frames)
            if frame_idx == e - 1:
                file = encoder.finish()
                target = Path(vstream_filename_format.format(s, e))
                temporary = target.with_suffix(".partial.hevc")
                temporary.write_bytes(file)
                temporary.replace(target)
                del file
                files.append(str(target))

                clip_idx += 1
                if clip_idx == len(clips):
                    break
                s, e = clips[clip_idx]
    assert clip_idx == len(clips) == len(files)

    decoder.finish()
    gc.collect()
    # torch.cuda.empty_cache()
    # nvcv.clear_cache()
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_clip_dir", type=str, required=True)
    parser.add_argument("--input_video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--width", type=int, default=7680, help="Width of the output video.")
    parser.add_argument("--height", type=int, default=3840, help="Height of the output video.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the output video.")
    parser.add_argument("--batch_size", type=int, default=1, help="BS of the Decoder, set to 1 for transcodeing.")
    parser.add_argument("--device_id", type=int, default=0, help="Specify the GPU ID if you have multiple GPUs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.output_dir is None:
        args.output_dir = args.input_clip_dir + "_vstreams"

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Using CUDA device: {args.device_id}.")

    if args.fps != 30 or args.batch_size != 1:
        parser.error("MUGEN requires --fps 30 and --batch_size 1")
    cuda.init()
    cuda_device = cuda.Device(args.device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    try:
        cvcuda_stream = cvcuda.Stream().current
        torch_stream = torch.cuda.default_stream(device=args.device_id)

        decoder = VideoBatchDecoder(
            args.width,
            args.height,
            args.fps,
            args.batch_size,
            args.device_id,
            cuda_ctx,
            cvcuda_stream,
        )
        assert decoder.fps == 30

        encoder = VideoMemoryEncoder(
            args.width,
            args.height,
            args.fps,
            args.batch_size,
            args.device_id,
            cuda_ctx,
            cvcuda_stream,
        )

        vids = sorted(p.stem for p in Path(args.input_clip_dir).glob("*.txt"))
        if not vids:
            parser.error("No clip metadata found")
        logging.info(f"Total {len(vids)} file(s) to process.")

        for idx, vid in enumerate(vids, start=1):
            clips = read_clips(Path(args.input_clip_dir) / f"{vid}.txt")

            logging.info(f"[{idx}/{len(vids)}] Start processing '{vid}'.")

            os.makedirs(os.path.join(args.output_dir, vid), exist_ok=True)

            files = process_one_video(
                str(find_video(args.input_video_dir, vid)),
                os.path.join(args.output_dir, vid, f"{vid}_{{:07d}}_{{:07d}}.hevc"),
                clips,
            )

            logging.info(f"Finish process {len(files)} clips of '{vid}'.")

    finally:
        cuda_ctx.pop()
