<div align="center">
  
# Tucano: Advancing Neural Text Generation for Portuguese

<a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-009C3B.svg" alt="arXiv">
</a>
<a href="https://huggingface.co/TucanoBR" target="_blank">
    <img src="https://img.shields.io/badge/HF%20Models-Tucano-FFDF00.svg" alt="HF Link">
</a>
<a href="https://github.com/Nkluge-correa/Tucano/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache-002776.svg" alt="License: MIT">
</a>

</div>
<p align="center">
        <img src="./img/logo.png" alt="An illustration of a Tucano bird showing vibrant colors like yellow, orange, blue, green, and black." height="400">
</p>

To stimulate the future of open development of neural text generation in Portuguese, we present both **GigaVerbo**, a concatenation of deduplicated Portuguese text corpora amounting to 200 billion tokens, and **Tucano**, a series of decoder-transformers based on the Llama 2 architecture, natively pre-trained in Portuguese. All byproducts of our study, including the source code used for training and evaluation, are openly released on GitHub and Hugging Face.

## Reproducing

This repository contains the source code used to train the Tucano series. We created all of our code implementations using a [`PyTorch`](https://github.com/pytorch/pytorch) based training script, while also using other auxiliary libraries to, for example, define our model's architecture ([`Transformers`](https://github.com/huggingface/transformers)), process our dataset ([`Datasets`](https://github.com/huggingface/datasets), [`Tokenizers`](https://github.com/huggingface/tokenizers), [`Sentencepiece`](https://github.com/google/sentencepiece)), optimize training speed and minimize memory footprint ([`Accelerate`](https://github.com/huggingface/accelerate), [`FlashAttention`](https://github.com/TriDao/FlashAttention), [`Triton`](https://github.com/triton-lang/triton), [`Liger Kernel`](https://github.com/linkedin/Liger-Kernel)), and logging of experiments ([`CodeCarbon`](https://github.com/mlco2/codecarbon), [`W&B`](https://github.com/wandb/wandb)). For DPO fine-tuning, we used [`TRL`](https://github.com/huggingface/trl).

All requirements are listed in the [`requirements.txt`](./requirements.txt) file. The Python version used is 3.10.12.

Given that our study was performed in the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin), which uses SLURM for job scheduling, all major scripts were launched via bash scripts. All bash scripts used carry the corresponding names of their corresponding Python scripts and can be found in the [`/scripts`](./scripts/) folder.

### Pretraining Corpus (GigaVerbo)

To learn more about the scripts used to handle our pretraining corpus, read [this file](./gigaverbo/README.md). To learn more about the filters used to parse GigaVerbo, read [this file](./gigaverbo/text-filter/README.md).

### Pretraining

Even though we repurposed the [TeenyTinyLlama](https://github.com/Nkluge-correa/TeenyTinyLlama) tokenizer for the Tucano models, for those interested in training new tokenizers, this repository contains two scripts for training tokenizers:

- Sentencepience ([`train-sentencepiece-tokenizer.py`](./train-sentencepiece-tokenizer.py)).
- BPE ([`train-BPE-tokenizer.py`](./train-BPE-tokenizer.py)).

The two primary scripts used to train the Tucano series are:

- The pretraining script ([`train-tucano.py`](./train-tucano.py)).
- The specifications script, where all arguments required for the pretraining script are defined ([`specifications.py`](./specifications.py)).

All training configurations are specified via YAML files (e.g., [`specs-tucano-1b1.yml`](./specs-tucano-1b1.yml)). You can learn more about the arguments passed via YAML files in this [`README`](./SPECIFICATIONS.md) file.

All logs (e.g., training logs, evaluation results, energy consumption) related to training and evaluating the Tucano series can be found in the [`/logs`](./logs) folder. The manipulation and creation of the plots associated with our study are done via the following [notebook](./logs/logs-and-plots.ipynb).

### Fine-Tuning

Our fine-tuning process was divided into two stages: supervised fine-tuning ("instruction-tuning") and direct preference optimization (DPO).

#### Supervised Fine-Tuning (SFT)

The main scripts used for SFT are:

- The supervised fine-tuning script ([`sft-tucano.py`](./sft-tucano.py)).
- Just like for the pretraining, all arguments required are defined via a YAML file (e.g., [`specs-tucano-sft.yml`](./specs-tucano-sft.yml)).

Read [this file](./cards/datasets/tucano-sft.md) to learn more about our supervised fine-tuning dataset.

#### Direct Preference Optimization

The main script used for the DPO-tuning are:

- The DPO script ([`dpo-tucano.py`](./dpo-tucano.py)).
- The `accelerate` configuration file ([`dpo-accelerate-config.yml`](./dpo-accelerate-config.yml))

To learn more about our preference modeling dataset, read [this dataset card](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset).

### Evaluations

To learn more about the evaluation harness used to access the capabilities of the Tucano series, read [this file](./evaluations/README.md).

## Intended Uses

The primary intended use of the Tucano models is to serve as foundations for research and development involving native Portuguese language modeling. Checkpoints saved during training are designed to provide a controlled setting for performing comparative experiments, specifically regarding the effects of active pretraining on the performance of currently available benchmarks. You may also fine-tune and adapt Tucano models for deployment if your use follows the Apache 2.0 license. If you decide to use the Tucano models as a basis for your fine-tuned model, please conduct your own risk and bias assessment.

## Out-of-scope Use

- Tucano models are **not intended for deployment**. They are not an out-of-the-box product and should not be used for human-facing interactions.

- Tucano models are for **the Portuguese language only** and are unsuitable for text generation tasks in other languages.

- Tucano models have **not been fine-tuned** for downstream tasks.

## Limitations

Like almost all other language models trained on large text datasets scraped from the web, the Tucano models show behavior that does not make them an out-of-the-box solution to many real-world applications, especially those requiring factual, reliable, and nontoxic text generation. Tucano models are all subject to the following:

- **Hallucinations:** Tucano models can produce content that can be mistaken as true facts, but are misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** Tucano models inherit the social and historical stereotypes from the data used to train them. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Unreliable Code:** Tucano models may produce incorrect code snippets and statements. These code generations should not be treated as suggestions or accurate solutions.

- **Language Limitations:** Tucano models are primarily designed to interact with Portuguese. Other languages might challenge its comprehension, leading to potential misinterpretations or errors in response.

- **Repetition and Verbosity:** Tucano models may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis on them if they intend to use them for real-world applications. We also have humans moderating the outputs of these models in applications where they will interact with an audience, guaranteeing users are always aware they are interacting with a language model.

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

Tucano is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](LICENSE) file.
