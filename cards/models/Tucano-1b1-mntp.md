---
language:
- pt
license: apache-2.0
library_name: transformers
tags:
- sentence-similarity
datasets:
- TucanoBR/GigaVerbo
- nicholasKluge/wikipedia-PT
pipeline_tag: sentence-similarity
co2_eq_emissions:
  emissions: 3690
  source: CodeCarbon
  training_type: pre-training
  geographical_location: Germany
  hardware_used: NVIDIA A100-SXM4-80GB
model-index:
- name: Tucano-1b1-mntp
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MassiveIntentClassification (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.5709
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MassiveScenarioClassification (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.6202
  - task:
      type: Classification
    dataset:
      type: mteb/multi-hatecheck
      name: MultiHateClassification (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.5859
  - task:
      type: Classification
    dataset:
      type: mteb/sib200
      name: SIB200Classification (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.6436
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_multilingual
      name: TweetSentimentClassification (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.4637
  - task:
      type: Classification
    dataset:
      type: hate-speech-portuguese/hate_speech_portuguese
      name: HateSpeechPortugueseClassification (pt)
      config: pt
      split: train
    metrics:
    - type: main_score
      value: 0.6096
  - task:
      type: Retrieval
    dataset:
      type: jinaai/mintakaqa
      name: MintakaRetrieval (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.0935
  - task:
      type: Retrieval
    dataset:
      type: ellamind/wikipedia-2023-11-retrieval-multilingual-queries
      name: WikipediaRetrievalMultilingual (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.5383
  - task:
      type: PairClassification
    dataset:
      type: nilc-nlp/assin2
      name: Assin2RTE (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.6383
  - task:
      type: STS
    dataset:
      type: mteb/stsb_multi_mt
      name: STSBenchmarkMultilingualSTS (pt)
      config: pt
      split: dev
    metrics:
    - type: main_score
      value: 0.6715
  - task:
      type: STS
    dataset:
      type: nilc-nlp/assin2
      name: Assin2STS (pt)
      config: pt
      split: test
    metrics:
    - type: main_score
      value: 0.5501
base_model:
- TucanoBR/Tucano-1b1
---
# Tucano-1b1-mntp

<img src="./logo.png" alt="An illustration of a Tucano bird showing vibrant colors like yellow, orange, blue, green, and black." height="200">

## Model Summary

**[Tucano](https://huggingface.co/TucanoBR)** is a series of decoder-transformers based on the Llama 2 architecture, pretrained natively in Portuguese. Tucano-1b1-mntp is a text-encoder generated via the methodology proposed in the [LLM2Vec](https://arxiv.org/abs/2404.05961) paper.

## Details

- **Architecture:** a decoder-only tranformer adapted for text-encoding tasks
- **Size:** 1,100,048,384 parameters
- **Context length:** 512 tokens
- **Dataset:** Pretraining: [TucanoBR/GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo) | Fine-tuning: [nicholasKluge/wikipedia-PT](https://huggingface.co/datasets/nicholasKluge/wikipedia-PT)
- **Language:** Portuguese
- **Number of steps:** 102,670
- **GPU:** 1 NVIDIA A100-SXM4-80GB
- **Training time**: ~ 27 hours
- **Emissions:** 3.69 KgCO2 (Germany)
- **Total energy consumption:** 9.99 kWh

This repository has the [source code](https://github.com/McGill-NLP/llm2vec) used to train this model.

### Training hyperparameters and results

The following hyperparameters were used during training (full configuration file available [here](https://huggingface.co/TucanoBR/Tucano-1b1-mntp/blob/main/tucano-mntb.json)):

- **learning rate:** 5e-05
- **train and evaluation batch size:** 32
- **seed:** 42
- **optimizer:** Adam with betas=(0.9,0.999) and epsilon=1e-08
- **scheduler type:** linear
- **epochs:** 3.0

A final validation perplexity of 4.71 and masked language modeling accuracy of 67% were achieved.

## Intended Uses

This model is only intended to be an experiment for the LLM2Vec methodology.

## Basic usage

```python
# pip install llm2vec
from llm2vec import LLM2Vec
import torch

l2v_model = LLM2Vec.from_pretrained(
        "TucanoBR/Tucano-1b1",
        peft_model_name_or_path="TucanoBR/Tucano-1b1-mntp",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

# Encoding queries using instructions
instruction = (
    "Dada uma consulta de pesquisa na Web, recupere trechos relevantes que respondam à consulta:"
)
queries = [
    [instruction, "Qual a linguagem de programação mais usada por cientistas de dados?"],
    [instruction, "Qual a capital do Brasil?"],
]
q_reps = l2v_model.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "Python é uma linguagem de programação de alto nível, interpretada de script, imperativa, orientada a objetos, funcional, de tipagem dinâmica e forte. Foi lançada por Guido van Rossum em 1991.",
    "Brasília é a capital do Brasil. Está localizada no Distrito Federal do Brasil, que possui pouco mais de 3,05 milhões de habitantes em uma área de 5802 quilômetros quadrados. O Distrito Federal consiste em um único município (município), Brasília, em que o município e o distrito federal são legalmente idênticos."
]
d_reps = l2v_model.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
#>>> tensor([[0.5717, 0.2799], [0.2542, 0.5236]])
```

## Evaluations

The table below provides evaluation results from several portuguese benchmarks from the [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb/tree/main).

| Model           | Average | PairClassification | STS    | Classification | Retrieval |
|-----------------|---------|--------------------|--------|----------------|-----------|
| Tucano-1b1-mntp | 0.5465  | 0.6381             | 0.6106 | 0.5824         | 0.3288    |

To reproduce these results, use the [`mteb-custom-eval.py`](https://huggingface.co/TucanoBR/Tucano-1b1-mntp/blob/main/mteb-custom-eval.py) script. Individual results can be found in the [`evals.json`](https://huggingface.co/TucanoBR/Tucano-1b1-mntp/blob/main/evals.json) file.

## Cite as 🤗

```latex
@misc{correa24tucano,
  title = {{Tucano: Advancing Neural Text Generation for Portuguese}},
  author = {Corr{\^e}a, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## Aknowlegments

We gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing \& Analytics Lab.

## License

Tucano is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](../../LICENSE) file.
