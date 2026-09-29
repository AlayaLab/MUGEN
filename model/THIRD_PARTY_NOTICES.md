# Third-party model components

Wan360.safetensors is a modified checkpoint based on Wan2.2-TI2V-5B, adapted for camera-controlled panoramic video generation. It is distributed under Apache-2.0.

| Component | Upstream source | License |
| --- | --- | --- |
| Wan360 base model and Wan2.2_VAE.pth | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | Apache-2.0 |
| models_t5_umt5-xxl-enc-bf16.pth | UMT5 encoder distributed with [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), based on Google UMT5 | Apache-2.0 |
| google/umt5-xxl/ tokenizer files | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) | Apache-2.0 |

The VAE, encoder and tokenizer are redistributed for inference; they are not presented as original MUGEN contributions. Upstream copyright and attribution notices remain applicable. A copy of Apache-2.0 is included in [LICENSE-WEIGHTS](LICENSE-WEIGHTS).
