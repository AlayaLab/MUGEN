"""Generate forward/backward/left/right translations from one ERP panorama."""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRECTIONS = {"forward": (0, 0, 1), "backward": (0, 0, -1),
              "left": (-1, 0, 0), "right": (1, 0, 0)}
PROMPT = (
    "A pedestrian walks forward along a wide stone pathway in a traditional Chinese garden, "
    "passing other visitors who stroll in the opposite direction. Weeping willow trees with "
    "long green branches line both sides, and sunlight filters through the leaves to create "
    "shifting dappled patterns on the grey stone paving. A stone-railed pond sits to the right."
)


def camera_trajectory(direction, num_frames, distance):
    import numpy as np
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    poses[:, :3, 3] = (np.linspace(0, distance, num_frames, dtype=np.float32)[:, None]
                       * np.asarray(DIRECTIONS[direction], dtype=np.float32))
    return poses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("WAN360_CKPT"),
                        help="Full Wan360 DiT checkpoint (or set WAN360_CKPT)")
    parser.add_argument("--base_dir", default=os.environ.get("WAN360_BASE_DIR"),
                        help="Directory containing Wan2.2_VAE.pth and T5 weights")
    parser.add_argument("--tokenizer_dir", default=os.environ.get("WAN360_TOKENIZER_DIR"),
                        help="Directory containing google/umt5-xxl/")
    parser.add_argument("--image", type=Path, default=ROOT / "example/first_frame.jpg")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/translations")
    parser.add_argument("--distance", type=float, default=1.6,
                        help="Total translation in camera-coordinate units, not calibrated meters")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=PROMPT,
                        help="Prompt template; {direction} is replaced for each video")
    args = parser.parse_args()
    for field in ("checkpoint", "base_dir", "tokenizer_dir"):
        if not getattr(args, field):
            parser.error(f"Set --{field} or its documented environment variable")
    if args.height <= 0 or args.width != 2 * args.height or args.height % 32:
        parser.error("Use a 2:1 panorama with height divisible by 32")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num_frames must be 4n+1 and at least 5")
    if not math.isfinite(args.distance) or args.distance <= 0 or args.steps <= 0:
        parser.error("--distance and --steps must be positive (distance must be finite)")
    return args


def main():
    args = parse_args()
    vae = Path(os.environ.get("WAN360_VAE_PATH", str(Path(args.base_dir) / "Wan2.2_VAE.pth")))
    t5 = Path(os.environ.get("WAN360_T5_PATH", str(Path(args.base_dir) / "models_t5_umt5-xxl-enc-bf16.pth")))
    tokenizer = Path(args.tokenizer_dir) / "google/umt5-xxl"
    for path in (args.image, Path(args.checkpoint), vae, t5):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)
    for direction in DIRECTIONS:
        for suffix in (".mp4", ".npz", ".json"):
            path = args.output_dir / (direction + suffix)
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}; choose another --output_dir")

    # Keep --help and argument validation independent of CUDA/model imports.
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    import numpy as np
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    with Image.open(args.image) as source:
        if source.width != 2 * source.height:
            raise ValueError("Input image must be a 2:1 ERP panorama")
        first_frame = source.convert("RGB").resize((args.width, args.height))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=str(args.checkpoint)),
                       ModelConfig(path=str(vae)), ModelConfig(path=str(t5))],
        tokenizer_config=ModelConfig(path=str(tokenizer)),
        redirect_common_files=False,
    )
    pipe.dit.is_360 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for direction in DIRECTIONS:
        poses = camera_trajectory(direction, args.num_frames, args.distance)
        prompt = args.prompt.replace("{direction}", direction)
        print(f"Generating {direction}: distance={args.distance}\nPrompt: {prompt}", flush=True)
        with torch.inference_mode():
            video = pipe(
                prompt=prompt,
                negative_prompt="模糊, 过曝, 字幕, 水印, 畸形, 低质量, 静态画面",
                input_image=first_frame.copy(), height=args.height, width=args.width,
                num_frames=args.num_frames, num_inference_steps=args.steps,
                cfg_scale=5.0, sigma_shift=5.0, seed=args.seed,
                tiled=True, is_360=True, cam_c2w=poses,
            )
        save_video(video, str(args.output_dir / f"{direction}.mp4"), fps=16, quality=5)
        np.savez(args.output_dir / f"{direction}.npz", data=poses)
        metadata = {
            "direction": direction, "distance": args.distance, "prompt": prompt,
            "seed": args.seed, "steps": args.steps, "width": args.width,
            "height": args.height, "num_frames": args.num_frames, "fps": 16,
            "is_360": True, "checkpoint": str(Path(args.checkpoint).resolve()),
            "image": str(args.image.resolve()),
        }
        with (args.output_dir / f"{direction}.json").open("x") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        del video


if __name__ == "__main__":
    main()
