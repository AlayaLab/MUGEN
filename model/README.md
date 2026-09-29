# Wan360

[Project overview](../README.md) · [MUGEN dataset](../dataset/README.md)

Generate panoramic videos with camera control from a single image. The inference entry [inference.py](inference.py) produces forward, backward, left and right camera translations.

## Installation

From `MUGEN/model/`, create a separate inference environment (NVIDIA GPU required):

```bash
conda create -n wan360-inference python=3.12 pip -y
conda activate wan360-inference
python -m pip install --no-deps torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip check
```

## Model weights

**Model repository:** [AlayaLab/Wan360](https://huggingface.co/AlayaLab/Wan360)

The repository includes the full step-3650 [Wan360 checkpoint](https://huggingface.co/AlayaLab/Wan360/resolve/main/Wan360.safetensors), `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, and the tokenizer under `google/umt5-xxl/`.

From `MUGEN/model/`, download the weights and set the paths:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('AlayaLab/Wan360', local_dir='weights/Wan360')"
export WAN360_CKPT="$PWD/weights/Wan360/Wan360.safetensors"
export WAN360_BASE_DIR="$PWD/weights/Wan360"
export WAN360_TOKENIZER_DIR="$PWD/weights/Wan360"
```

VAE, T5 and tokenizer files are all included in this download. Override `WAN360_VAE_PATH` and `WAN360_T5_PATH` if needed. Run inference with the code in this repository.

## Inference

From `MUGEN/model/`, with the environment activated and weight paths set:

```bash
python inference.py
```

The script uses [example/first_frame.jpg](example/first_frame.jpg) and generates four videos in `outputs/translations/`: `forward.mp4`, `backward.mp4`, `left.mp4`, and `right.mp4`. Each starts from the same image and seed, with fixed camera orientation and constant-speed translation. No CSV or external camera annotations are needed.

### Default prompt

The default prompt is:

> A pedestrian walks forward along a wide stone pathway in a traditional Chinese garden, passing other visitors who stroll in the opposite direction. Weeping willow trees with long green branches line both sides, and sunlight filters through the leaves to create shifting dappled patterns on the grey stone paving. A stone-railed pond sits to the right.

All four directions use this same prompt; the camera trajectory specifies the movement direction.

### Camera motion

The default camera amplitude is **`--distance 1.6`**, the total displacement over the whole video. With **161 frames**, each step translates **0.01** camera-coordinate units. These units are not calibrated meters. Every trajectory starts at the origin and keeps an identity rotation (no camera rotation).

| Direction | Final camera position `(x, y, z)` |
| --- | --- |
| Forward | `(0, 0, 1.6)` |
| Backward | `(0, 0, -1.6)` |
| Left | `(-1.6, 0, 0)` |
| Right | `(1.6, 0, 0)` |

Forward (+Z) points toward the panorama's horizontal center. Changing `--num_frames` keeps the total displacement at 1.6 unless `--distance` is also changed.

### Generation settings

Defaults: **1920×960, 161 frames at 16 FPS, 50 steps, seed 42**. For example:

```bash
python inference.py --distance 1.6 --output_dir outputs/example_run
```

Use `python inference.py --help` for options. Existing outputs are not overwritten.

## Outputs

Each direction produces three files under the selected output directory:

| File | Contents |
| --- | --- |
| `<direction>.mp4` | Generated panoramic video |
| `<direction>.npz` | Camera-to-world trajectory, key `data`, shape `[num_frames, 4, 4]` |
| `<direction>.json` | Prompt and generation settings for that video |

Generated outputs remain local and are excluded from Git.

## License

Wan360 weights are licensed under [Apache-2.0](LICENSE-WEIGHTS). Model code retains its [Apache-2.0 license](LICENSE). Model code is based on DiffSynth-Studio. The checkpoint adapts Wan2.2-TI2V-5B; bundled VAE, UMT5 encoder and tokenizer retain their upstream licenses and notices. See [third-party notices](THIRD_PARTY_NOTICES.md).

The [example photograph](example/README.md) was captured by the MUGEN team and is separately licensed under [CC BY 4.0](example/LICENSE).
