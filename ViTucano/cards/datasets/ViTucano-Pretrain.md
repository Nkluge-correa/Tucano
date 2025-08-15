---
license: other
language:
  - pt
pretty_name: ViTucano-Pretrain
task_categories:
  - image-to-text
  - text-generation
size_categories:
  - 100K<n<1M
viewer: false
tags:
  - image-to-text
---

# ViTucano-Pretrain

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
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Aknowlegments](#aknowlegments)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain
- **Repository:** https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain
- **Paper:** [Tucano: Advancing Neural Text Generation for Portuguese](https://arxiv.org/abs/2411.07854)
- **Point of Contact:** [Nk-correa](mailto:kluge@uni-bonn.de)

### Dataset Summary

ViTucano-Pretrain is a translation of the original [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), obtained via Google's translation API. LLaVA Visual Instruct Pretrain LCS-558K is a subset of the LAION/CC/SBU dataset, filtered with a more balanced concept coverage distribution. This dataset was used to train the **ViTucano**, our first attempt at creating a vision assistant natively pretrained in Portuguese. **ViTucano** is built on top of the [Tucano series](https://arxiv.org/abs/2411.07854) using the [TinyLLaVA Factory](https://arxiv.org/abs/2405.11788).

### Supported Tasks and Leaderboards

This dataset can be utilized for tasks involving language modeling and visual instruction tunning.

### Languages

Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **id:** an identifier (name of the respective file) for that image.
- **image:** the path to the file in the original folder configuration.
- **conversations:** a list of dictionaries, where each dictionary represents a message or an entry in a conversation.
- **blip_caption:** the original BLIP caption.
- **url:** the url of the corresponding image.

### Data Fields

```python
{
    "id": "004539375",
    "image": "train/00453/004539375.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "Renderize um resumo claro e conciso da foto.\n<image>"
        },
        {
            "from": "gpt",
            "value": "Selecione móveis de luxo 3 - colchão de espuma de memória de gel de polegada"
        }
    ],
    "blip_caption": "Selecione móveis de luxo 3 - colchão de espuma de memória de gel de polegada",
    "url": "http://ec1.ostkcdn.com/images/products/8111140/P15459545.jpg"
}
```

### Data Splits

Available splits are `train`.

To use this dataset, you will need to download both the `data-pretraining.json` and `images.zip` files available in this folder:

```bash
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/data-pretraining.json
wget https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain/resolve/main/images.zip
```

You can also do this via the `huggingface_hub` library:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="ViTucano-Pretrain", repo_type="dataset")
```

Unzip the images in a way that you get this folder structure (e.g., `unzip images.zip -d "path/to/train"`):

```bash
├── train
    ├── 00000
    ├── 00001
    ├── 00002
    └── etc ...
```

Done! The data is ready to train your projector.

## Dataset Creation

### Curation Rationale

This dataset is a translation of the original [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) obtained via Google's translation API.

### Source Data

#### Who are the source language producers?

All text samples translated from English to Portuguese.

### Annotations

#### Annotation process

Read this [dataset card](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) for more information.

#### Who are the annotators?

Read this [dataset card](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) for more information.

### Considerations for Using the Data

**Warning:** This dataset may contain NSFW (Not Safe For Work) content, including explicit images and text captions with offensive/sensitive language.

### Other Known Limitations

This dataset has has been translated using translation engines, potentially resulting in corrupted samples. While useful for quickly converting text between languages, translation engines often struggle with accurately preserving the syntax, semantics, and context of certain languages.

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:kluge@uni-bonn.de).

### Licensing Information

Users of this dataset must comply with license of [CC-3M](https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE) and [BLIP](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) (if you use their synthetic caption).

Creative Commons Attribution 4.0 International; and it should abide by the [policy of OpenAI](https://openai.com/policies/terms-of-use).

### Citation Information

#### ViTucano

```bibtex
@misc{correa2025vitucano,
    author={Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
    title={{ViTucano: A Portuguese Vision Assistant}},
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

@article{correa2025tucanoadvancingneuraltext,
    title={{Tucano: Advancing Neural Text Generation for Portuguese}},
    author={Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
    journal={Patterns},
    publisher={Elsevier},
    year={2025},
    doi={10.1016/j.patter.2025.101325},
    url={https://doi.org/10.1016/j.patter.2025.101325},
    issn={2666-3899}
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
