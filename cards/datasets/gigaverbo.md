---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: int64
    class_label:
        names:
          '0': low
          '1': high
  - name: probs
    dtype: float64    
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 786084805068
    num_examples: 145300844
  download_size: 411184278869
  dataset_size: 786084805068
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: other
task_categories:
- text-generation
language:
- pt
tags:
- portuguese
- language-modeling
pretty_name: GigaVerbo
size_categories:
- 100M<n<1B
---

# GigaVerbo: a 780 GB Dataset of Portuguese Text

<img src="./logo-gigaverbo.png" height="200">

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
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Aknowlegments](#aknowlegments)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://huggingface.co/datasets/TucanoBR/GigaVerbo
- **Repository:** https://huggingface.co/datasets/TucanoBR/GigaVerbo
- **Paper:** [Tucano: Advancing Neural Text Generation for Portuguese](https://arxiv.org/abs/xxxx.xxxxx)
- **Point of Contact:** [Nk-correa](mailto:kluge@uni-bonn.de)

### Dataset Summary

GigaVerbo is an extensive dataset comprising **780 GB of Portuguese text**, being a concatenated version of several datasets available in [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:text-generation&language=language:pt&sort=trending), containing over **200 billion tokens**. It encompasses various sources, including crawled websites, articles, translated conversations, and legal documents. This dataset offers a comprehensive and rich resource for various natural language processing tasks, providing researchers and developers with ample material for training and testing language models, text analysis algorithms, and other language-related applications for Portuguese. This dataset was used to train the Tucano series, described in "_[Tucano: Advancing Neural Text Generation for Portuguese](https://arxiv.org/abs/xxxx.xxxxx)_".

### Supported Tasks and Leaderboards

This dataset can be utilized for tasks involving language modeling.

### Languages

Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **text:** a string of text in Portuguese.
- **metadata:** the source where that string originated.
- **label:** the class label assined by [TucanoBR/BERTimbau-base-text-filter](https://huggingface.co/TucanoBR/BERTimbau-base-text-filter) to the corresponding `text` string (1 = high, 0 = low).
- **probs:** the confidence score assigned to the corresponding `label`.

### Data Fields

```python
{
  "text": "A inteligência artificial (de sigla: IA; do inglês: artificial intelligence, de sigla: AI) é um campo de estudo multidisciplinar que abrange varias áreas do conhecimento ...",
  "metadata": "source: https://huggingface.co/datasets/graelo/wikipedia",
  "label": 1,
  "probs" : 0.99
}
```

### Data Splits

Available splits are `train`.

```python
from datasets import load_dataset

dataset = load_dataset("TucanoBR/GigaVerbo", split='train')

# If you don't want to download the entire dataset, set streaming to `True`
dataset = load_dataset("TucanoBR/GigaVerbo", split='train', streaming=True)

```

## Dataset Creation

### Curation Rationale

This dataset was developed as part of the study "[Tucano: Advancing Neural Text Generation for Portuguese](https://arxiv.org/abs/xxxx.xxxxx)". In short, GigaVerbo is the concatenation of several [openly available Portuguese text datasets](https://huggingface.co/datasets?task_categories=task_categories:text-generation&language=language:pt&sort=trending).

### Source Data

#### Initial Data Collection and Normalization

GigaVerbo has been deduplicated with an [exact hash deduplication filter](https://github.com/ChenghaoMou/text-dedup) and filtered by [TucanoBR/BERTimbau-base-text-filter](https://huggingface.co/TucanoBR/BERTimbau-base-text-filter). However, all examples classified as low quality still reside in this original dataset. We leave the task of parsing GigaVerbo concerning class label and confidence of the used classifier to the user so that one can tune this filtering as they see fit.

A class label distribution of the samples in GigaVerbo can be found in the table below:

| Subset          | Original Size   | High           | Low            |
|-----------------|-----------------|----------------|----------------|
| monoHPLT-PT     | 58,244,012      | 33,650,933     | 24,593,079     |
| CrawlPT         | 43,846,974      | 27,498,861     | 16,348,113     |
| Multilingual-C4 | 16,092,571      | 13,440,818     | 2,651,753      |
| Common Crawl    | 12,470,998      | 10,073,993     | 2,397,005      |
| BlogSet-BR      | 4,321,181       | 2,064,925      | 2,256,256      |
| Instruct-PTBR   | 2,962,856       | 2,454,851      | 508,005        |
| Corpus Carolina | 2,075,395       | 1,097,758      | 977,637        |
| UltrachatBR     | 1,255,091       | 1,244,349      | 10,742         |
| Wikipedia       | 1,101,475       | 897,264        | 204,211        |
| CulturaX        | 999,994         | 855,725        | 144,269        |
| LegalPT         | 925,522         | 856,814        | 68,708         |
| Gpt4All         | 808,803         | 685,159        | 123,644        |
| Bactrian-X      | 66,994          | 52,764         | 14,230         |
| XL-SUM          | 64,577          | 64,376         | 201            |
| Dolly 15K       | 28,401          | 19,643         | 8,758          |
| CosmosQA        | 25,260          | 11,810         | 13,450         |
| ROOTS           | 10,740          | 4,911          | 5,829          |
| **Total**       | **145,300,844** | **94,974,954** | **50,325,890** |

#### Who are the source language producers?

All text samples are native to Portuguese or translated from other languages to Portuguese (slight contamination of different languages should also be expected).

### Annotations

#### Annotation process

GigaVerbo is the concatenation of several [openly available Portuguese text datasets](https://huggingface.co/datasets?task_categories=task_categories:text-generation&language=language:pt&sort=trending).

#### Who are the annotators?

[Nicholas Kluge Corrêa](mailto:kluge@uni-bonn.de).

### Personal and Sensitive Information

This dataset can potentially contain personal and sensitive information, along with offensive, toxic, and disturbing language.

## Considerations for Using the Data

### Social Impact of Dataset

The presence of personal and sensitive information within the dataset raises concerns about privacy and data protection, potentially leading to breaches of individuals' confidentiality and security. Furthermore, the inclusion of offensive, toxic, and disturbing language in the dataset poses risks of perpetuating harmful behaviors and attitudes, contributing to the normalization of hate speech and online toxicity. Therefore, careful handling and ethical considerations are essential to mitigate these potential social impacts and promote responsible dataset use.

### Discussion of Biases

The inclusion of offensive, toxic, and disturbing language in the dataset poses risks of perpetuating harmful behaviors and attitudes, contributing to the normalization of hate speech and online toxicity.

### Other Known Limitations

A significant portion of the dataset's data has been translated using translation engines, potentially resulting in corrupted samples of both language and code. While useful for quickly converting text between languages, translation engines often struggle with accurately preserving the syntax, semantics, and context of programming languages. As a result, the translated code may contain errors, syntax inconsistencies, or even introduce vulnerabilities, rendering it unreliable or unusable for its intended purpose.

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:kluge@uni-bonn.de).

### Licensing Information

The following datasets and respective licenses from GigaVerbo (only training splits are a part of the corpus):

- [HPLT-PT](https://huggingface.co/datasets/HPLT/hplt_monolingual_v1_2) (License: [cc0-1.0](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information))

- [CC-2023](https://huggingface.co/datasets/dominguesm/CC-MAIN-2023-23) (License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en))

- [CCc100](https://huggingface.co/datasets/eduagarcia/CrawlPT_dedup) (License: [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/))

- [MC4-PT](https://huggingface.co/datasets/thegoodfellas/mc4-pt-cleaned) (License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html))

- [Blogset-BR](https://huggingface.co/datasets/thegoodfellas/blogset-br) (License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html))

- [BrWaC](https://huggingface.co/datasets/UFRGS/brwac) (License: Unknown)

- [Instruct-PTBR](https://huggingface.co/datasets/cnmoro/Instruct-PTBR-ENUS-11M) (License: [LLAMA 2 Community License](https://ai.meta.com/llama/license/))

- [Wikipedia](https://huggingface.co/datasets/graelo/wikipedia) (License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

- [Corpus Carolina](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina) (License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en))

- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) (License: [ODC-By](https://opendatacommons.org/licenses/by/1-0/), [cc0-1.0](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information))

- [Gpt4all](https://huggingface.co/datasets/pablo-moreira/gpt4all-j-prompt-generations-pt) (License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html))

- [UltrachatBR](https://huggingface.co/datasets/recogna-nlp/UltrachatBR) (License: [MIT](https://mit-license.org/))

- [OSCAR](https://huggingface.co/datasets/eduagarcia/CrawlPT_dedup) (License: [cc0-1.0](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information))

- [Legal Portuguese](https://huggingface.co/datasets/eduagarcia/LegalPT_dedup) (License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en))

- [Xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum) (License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en))

- [Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) (License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.de))

- [Dolly-15k](https://huggingface.co/datasets/Gustrd/dolly-15k-libretranslate-pt) (License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

- [CosmosQA](https://huggingface.co/datasets/heloisy/cosmos_qa_ptbr) (License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.de))

- [Roots Wikiquote](https://huggingface.co/datasets/bigscience-data/roots_pt_wikiquote) (License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

- [Roots Ted Talks](https://huggingface.co/datasets/bigscience-data/roots_pt_ted_talks_iwslt) (License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en))

### Citation Information

```latex

@misc{correa24tucano,
  title = {{Tucano: Advancing Neural Text Generation for Portuguese}},
  author = {Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}

```

### Aknowlegments

We gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing \& Analytics Lab.

### Contributions

If you want to contribute, contact me at [kluge@uni-bonn.de](mailto:kluge@uni-bonn.de)!
