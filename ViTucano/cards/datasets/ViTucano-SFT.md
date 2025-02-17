---
license: other
task_categories:
- image-to-text
- text-generation
language:
- pt
pretty_name: ViTucano-SFT
size_categories:
- 100K<n<1M
viewer: false
tags:
- image-to-text
---
# ViTucano-SFT

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Known Limitations](#known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Aknowlegments](#aknowlegments)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://huggingface.co/datasets/TucanoBR/ViTucano-SFT
- **Repository:** https://huggingface.co/datasets/TucanoBR/ViTucano-SFT
- **Paper:** [Tucano: Advancing Neural Text Generation for Portuguese](https://arxiv.org/abs/2411.07854)
- **Point of Contact:** [Nk-correa](mailto:kluge@uni-bonn.de)

### Dataset Summary

ViTucano-SFT is a dataset for visual instruction tuning. To built it, we used samples from the original [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) dataset. More specifically, only the samples from the `coco` and `gqa` partitions are needed. These samples were then translated into Portuguese using Google's translation API. The original dataset ([LLaVA Visual Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)) was created by prompting GPT-4-0314 API. We also added samples (i.e., the entire train portion) from the [COCO Captions Portuguese Translation](https://huggingface.co/datasets/laicsiifes/coco-captions-pt-br). This dataset was used to train the **ViTucano**, our first attempt at creating a vision assistant natively pretrained in Portuguese. **ViTucano** is built on top of the [Tucano series](https://arxiv.org/abs/2411.07854) using the [TinyLLaVA Factory](https://arxiv.org/abs/2405.11788).

### Supported Tasks and Leaderboards

This dataset can be utilized for tasks involving language modeling and visual instruction tunning.

### Languages

Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **image:** the path to the file in the original folder configuration.
- **id:** an identifier (name of the respective file) for that image.
- **conversations:** a list of dictionaries, where each dictionary represents a message or an entry in a conversation.
- **partition:** the original dataset that this sample comes from (e.g., coco, gqa, or coco-captions-pt-br).

### Data Fields

```python
{
    "id": "000000444448",
    "image": "train/coco_1/000000444448.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Quem está pintando o hidrante na imagem?\n<image>"
      },
      {
        "from": "gpt",
        "value": "Na imagem, uma mulher está pintando o hidrante em uma calçada."
      }
    ],
    "partition": "coco_1"
}
```

### Data Splits

Available splits are `train`.

To use this dataset, you will need to download both the `data-sft.json` file and all corresponding zip files available in this folder (`coco_1.zip`, `coco_2.zip`, `coco-captions-pt-br.zip`, and `gqa.zip` ):

```bash
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/data-sft.json
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/coco_1.zip
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/coco_2.zip
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/coco-captions-pt-br.zip
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/gqa.zip
```

You can also do this via the `huggingface_hub` library:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="ViTucano-SFT", repo_type="dataset")
```

Unzip the images in a way that you get this folder structure (e.g., `unzip coco_1.zip -d "path/to/train"`):

```bash
├── train
    ├── coco_1
    ├── coco_2
    ├── coco-captions-pt-br
    └── gqa
```

Done! The data is ready to fine-tune your visual assitant.

## Dataset Creation

### Curation Rationale

This dataset has samples from the original [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) dataset. More specifically, only the samples from the `coco` and `gqa` partitions are needed. These samples were then translated into Portuguese using Google's translation API. The original dataset ([LLaVA Visual Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)) was created by prompting GPT-4-0314 API. We also added samples (i.e., the entire train portion) from the [COCO Captions Portuguese Translation](https://huggingface.co/datasets/laicsiifes/coco-captions-pt-br).

### Source Data

#### Who are the source language producers?

All text samples translated from English to Portuguese.

### Annotations

#### Annotation process

Read this [dataset card](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) for more information.

#### Who are the annotators?

Read this [dataset card](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) for more information.

### Known Limitations

This dataset has has been translated using translation engines, potentially resulting in corrupted samples. While useful for quickly converting text between languages, translation engines often struggle with accurately preserving the syntax, semantics, and context of certain languages.

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:kluge@uni-bonn.de).

### Licensing Information

Creative Commons Attribution 4.0 International; and it should abide by the [policy of OpenAI](https://openai.com/policies/terms-of-use).

### Citation Information

#### ViTucano

```bibtex
@misc{correa2025vitucano,
    author={Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
    title={{ViTucano: A Portuguese Vision Assitant}},
    year=2025,
    howpublished={\url{https://huggingface.co/TucanoBR/ViTucano-2b8-v1}},
    doi={10.57967/hf/4530},
    publisher={{Hugging Face}}
}
```

#### Tucano

```bibtex
@misc{correa2024tucanoadvancingneuraltext,
      title={{Tucano: Advancing Neural Text Generation for Portuguese}}, 
      author={Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
      year={2024},
      eprint={2411.07854},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.07854}, 
}
```

#### TinyLLaVA Factory

```bibtex
@article{jia2024tinyllava,
  title={TinyLLaVA Factory: A Modularized Codebase for Small-scale Large Multimodal Models},
  author={Jia, Junlong and Hu, Ying and Weng, Xi and Shi, Yiming and Li, Miao and Zhang, Xingjian and Zhou, Baichuan and Liu, Ziyu and Luo, Jie and Huang, Lei and Wu, Ji},
  journal={arXiv preprint arXiv:2405.11788},
  year={2024}
}
```

#### LLaVA

```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

### Aknowlegments

We gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing \& Analytics Lab.

### Contributions

If you want to contribute, contact me at [kluge@uni-bonn.de](mailto:kluge@uni-bonn.de)!
