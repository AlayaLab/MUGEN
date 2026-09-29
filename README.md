<h1 align="center">MUGEN: Interactive Panoramic World Exploration via Camera Control</h1>

<p align="center"><b>SIGGRAPH Asia 2026</b></p>

<p align="center">
  <a href="https://alaya-lab.github.io/MUGEN/"><img src="https://img.shields.io/badge/Project_Page-MUGEN-1a73e8.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2609.38077"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/AlayaLab/MUGEN"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Dataset-AlayaLab%2FMUGEN-ffce1c.svg" alt="Dataset"></a>
  <a href="https://huggingface.co/AlayaLab/Wan360"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Weights-AlayaLab%2FWan360-ffce1c.svg" alt="Model Weights"></a>
</p>

![MUGEN overview: panoramic videos with camera trajectories, instance masks, depth maps and semantic annotations](https://alaya-lab.github.io/MUGEN/assets/overview.webp)

MUGEN brings together a large-scale real-world panoramic video dataset and **Wan360**, a model for generating immersive 360° videos with camera control from a single panoramic image. The dataset provides semantic and geometric annotations, while Wan360 adapts a perspective video backbone to panoramic geometry and user-specified camera trajectories.

## MUGEN dataset

The project reports **1,318 hours** of curated panoramic footage with **at least 4K** source resolution, organized into **60-second clips**, and a **300-hour MUGEN-HQ** training subset. Annotations cover captions, scene attributes, locations, camera trajectories, instance masks and depth.

![MUGEN collection, annotation, filtering and subset sampling pipeline](https://alaya-lab.github.io/MUGEN/assets/dataset.webp)

The current CSV release contains **77,023 clips (approximately 1,283.7 hours)**, including **17,655 HQ clips**. Captions and category annotations are included in the CSVs; camera trajectories are available on Hugging Face. Depth is pending release. See the [dataset guide](dataset/README.md) for the current release contents and reconstruction workflow.

## Wan360 model

Wan360 uses three parameter-free components—**periodic longitude RoPE**, **ERP-aware padding** and **random roll yaw**—to handle cyclic seams and yaw consistency in equirectangular projection (ERP). A **panoramic Plücker embedding** encodes camera motion using spherical ERP rays.

![Wan360 panoramic video generation method](https://alaya-lab.github.io/MUGEN/assets/method.webp)

The [project page](https://alaya-lab.github.io/MUGEN/) demonstrates forward, backward, left and right camera translations from the same input panorama. Follow the [model guide](model/README.md) to run inference.

## Getting started

| Component | Guide | What you can do |
| --- | --- | --- |
| **Wan360 model** | [Model guide](model/README.md) | Set up inference and generate forward, backward, left and right camera translations. |
| **MUGEN dataset** | [Dataset guide](dataset/README.md) | Access annotations, download source videos and reconstruct clips following the Sekai workflow. |

```text
MUGEN/
├── model/       # Wan360 inference code and example image
├── dataset/     # Dataset download, reconstruction and annotation tools
└── README.md    # Project overview
```

Each guide specifies its own environment and working directory.

## Resources

| Resource | Link / status |
| --- | --- |
| Project page and demos | [Interactive Panoramic World Exploration via Camera Control](https://alaya-lab.github.io/MUGEN/) |
| Paper | [arXiv](https://arxiv.org/abs/2609.38077) |
| Dataset and annotations | [AlayaLab/MUGEN on Hugging Face](https://huggingface.co/datasets/AlayaLab/MUGEN) |
| Wan360 weights | [AlayaLab/Wan360 on Hugging Face](https://huggingface.co/AlayaLab/Wan360) |

## Acknowledgements and licenses

Dataset tools are adapted from [Sekai](https://github.com/Lixsp11/sekai-codebase). Model code is based on [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio).

- **Dataset code:** [dataset/LICENSE](dataset/LICENSE), including its non-commercial research restriction.
- **Model code:** [model/LICENSE](model/LICENSE), Apache-2.0.
- **Example photograph:** Captured by the MUGEN team and released under [CC BY 4.0](model/example/LICENSE).
- **Wan360 weights:** [model/LICENSE-WEIGHTS](model/LICENSE-WEIGHTS), Apache-2.0.
- **MUGEN annotations:** [dataset/LICENSE-ANNOTATIONS](dataset/LICENSE-ANNOTATIONS), CC BY 4.0, including captions, category annotations, camera trajectories and depth when released.

The annotation license covers only rights held by the MUGEN contributors in the annotations and database; it does not license source videos or reconstructed video clips. Third-party assets retain their original licenses and notices. See [model/THIRD_PARTY_NOTICES.md](model/THIRD_PARTY_NOTICES.md) for the bundled model components.

## Citation

```bibtex
@misc{tan2026mugeninteractivepanoramicworld,
      title={MUGEN: Interactive Panoramic World Exploration via Camera Control},
      author={Jiaming Tan and Zhen Li and Shuwei Shi and Minggui Teng and Siqi Yang and Yuwei Wu and Bo Zheng and Chuanhao Li and Kaipeng Zhang},
      year={2026},
      eprint={2609.38077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.38077},
}
```
