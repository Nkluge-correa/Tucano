---
language:
- pt
license: apache-2.0
library_name: transformers
tags:
- text-generation-inference
datasets:
- TucanoBR/GigaVerbo
metrics:
- perplexity
pipeline_tag: text-generation
widget:
- text: "A floresta da Amazônia é conhecida por sua"
  example_title: Exemplo
- text: "Uma das coisas que Portugal, Angola, Brasil e Moçambique tem em comum é o"
  example_title: Exemplo
- text: "O Carnaval do Rio de Janeiro é"
  example_title: Exemplo
inference:
  parameters:
    repetition_penalty: 1.2
    temperature: 0.1
    top_k: 50
    top_p: 1.0
    max_new_tokens: 150
co2_eq_emissions:
  emissions: 4475000
  source: CodeCarbon
  training_type: pre-training
  geographical_location: Germany
  hardware_used: NVIDIA A100-SXM4-80GB
model-index:
- name: Tucano-2b4
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: CALAME-PT
      type: NOVA-vision-language/calame-pt
      split: all
      args:
        num_few_shot: 0
    metrics:
    - type: acc
      value: 59.06
      name: accuracy
    source:
      url: https://huggingface.co/datasets/NOVA-vision-language/calame-pt
      name: Context-Aware LAnguage Modeling Evaluation for Portuguese
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: LAMBADA-PT
      type: TucanoBR/lambada-pt
      split: train
      args:
        num_few_shot: 0
    metrics:
    - type: acc
      value: 37.67
      name: accuracy
    source:
      url: https://huggingface.co/datasets/TucanoBR/lambada-pt
      name: LAMBADA-PT
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: ENEM Challenge (No Images)
      type: eduagarcia/enem_challenge
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 20.5
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BLUEX (No Images)
      type: eduagarcia-temp/BLUEX_without_images
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 23.23
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: OAB Exams
      type: eduagarcia/oab_exams
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 25.47
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Assin2 RTE
      type: assin2
      split: test
      args:
        num_few_shot: 15
    metrics:
    - type: f1_macro
      value: 56.27
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Assin2 STS
      type: eduagarcia/portuguese_benchmark
      split: test
      args:
        num_few_shot: 10
    metrics:
    - type: pearson
      value: 1.93
      name: pearson
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: FaQuAD NLI
      type: ruanchaves/faquad-nli
      split: test
      args:
        num_few_shot: 15
    metrics:
    - type: f1_macro
      value: 43.97
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HateBR Binary
      type: ruanchaves/hatebr
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: f1_macro
      value: 29.49
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: PT Hate Speech Binary
      type: hate_speech_portuguese
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: f1_macro
      value: 41.98
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: tweetSentBR
      type: eduagarcia-temp/tweetsentbr
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: f1_macro
      value: 58.0
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: ARC-Challenge (PT)
      type: arc_pt
      args:
        num_few_shot: 25
    metrics:
    - type: acc_norm
      value: 30.43
      name: normalized accuracy
    source:
      url: https://github.com/nlp-uoregon/mlmm-evaluation
      name: Evaluation Framework for Multilingual Large Language Models
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag (PT)
      type: hellaswag_pt
      args:
        num_few_shot: 10
    metrics:
    - type: acc_norm
      value: 47.17
      name: normalized accuracy
    source:
      url: https://github.com/nlp-uoregon/mlmm-evaluation
      name: Evaluation Framework for Multilingual Large Language Models
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: TruthfulQA
      type: truthfulqa_pt
      args:
        num_few_shot: 0
    metrics:
    - type: mc2
      value: 39.3
      name: bleurt
    source:
      url: https://github.com/nlp-uoregon/mlmm-evaluation
      name: Evaluation Framework for Multilingual Large Language Models
---
# Tucano-2b4

<img src="./logo.png" alt="An illustration of a Tucano bird showing vibrant colors like yellow, orange, blue, green, and black." height="200">

## Model Summary

**[Tucano](https://huggingface.co/TucanoBR)** is a series of decoder-transformers natively pretrained in Portuguese. All Tucano models were trained on **[GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo)**, a concatenation of deduplicated Portuguese text corpora amounting to 200 billion tokens.

Read our preprint [here](https://arxiv.org/abs/2411.07854).

## Details

- **Architecture:** a Transformer-based model pre-trained via causal language modeling
- **Size:** 2,444,618,240 parameters
- **Context length:** 4096 tokens
- **Dataset:** [TucanoBR/GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo)
- **Language:** Portuguese
- **Number of steps:** 1,960,000
- **GPU:** 16 NVIDIA A100-SXM4-80GB
- **Training time**: ~ 845 hours
- **Emissions:** 4,475 KgCO2 (Germany)
- **Total energy consumption:** 11,749 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Tucano) used to train this model. The main libraries used are:

- [PyTorch](https://github.com/pytorch/pytorch)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [Tokenizers](https://github.com/huggingface/tokenizers)
- [Sentencepiece](https://github.com/google/sentencepiece)
- [Accelerate](https://github.com/huggingface/accelerate)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [Codecarbon](https://github.com/mlco2/codecarbon)
- [TRL](https://github.com/huggingface/trl)

## Intended Uses

The primary intended use of the Tucano models is to serve as foundations for research and development involving native Portuguese language modeling. Checkpoints saved during training are designed to provide a controlled setting for performing comparative experiments, specifically regarding the effects of active pretraining on the performance of currently available benchmarks. You may also fine-tune and adapt Tucano models for deployment if your use follows the Apache 2.0 license. If you decide to use the Tucano models as a basis for your fine-tuned model, please conduct your own risk and bias assessment.

## Out-of-scope Use

- Tucano models are **not intended for deployment**. They are not an out-of-the-box product and should not be used for human-facing interactions.

- Tucano models are for **the Portuguese language only** and are unsuitable for text generation tasks in other languages.

- Tucano models have **not been fine-tuned** for downstream tasks.

## Basic usage

Using the `pipeline`:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="TucanoBR/Tucano-2b4")

completions  = generator("A floresta da Amazônia é conhecida por sua", num_return_sequences=2, max_new_tokens=100)

for comp in completions:
  print(f"🤖 {comp['generated_text']}")
```

Using the `AutoTokenizer` and `AutoModelForCausalLM`:

```python
from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the model and tokenizer
model_id = "TucanoBR/Tucano-2b4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Specify the generation parameters as you like
generation_config = GenerationConfig(
    **{
    "do_sample": True,
    "max_new_tokens": 2048,
    "renormalize_logits": True,
    "repetition_penalty": 1.2,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 1.0,
    "use_cache": True, 
  }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = TextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, device=device)

# Generate text
prompt = "A floresta da Amazônia é conhecida por sua"
completion = generator(prompt, generation_config=generation_config)
print(completion[0]['generated_text'])
```

## Limitations

Like almost all other language models trained on large text datasets scraped from the web, the Tucano models show behavior that does not make them an out-of-the-box solution to many real-world applications, especially those requiring factual, reliable, and nontoxic text generation. Tucano models are all subject to the following:

- **Hallucinations:** Tucano models can produce content that can be mistaken as true facts, but are misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** Tucano models inherit the social and historical stereotypes from the data used to train them. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Unreliable Code:** Tucano models may produce incorrect code snippets and statements. These code generations should not be treated as suggestions or accurate solutions.

- **Language Limitations:** Tucano models are primarily designed to interact with Portuguese. Other languages might challenge its comprehension, leading to potential misinterpretations or errors in response.

- **Repetition and Verbosity:** Tucano models may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis on them if they intend to use them for real-world applications.

## Evaluations

The table below compares our models against several Portuguese and multilingual language models on the evaluation harness used in our study. More information on it can be found [here](https://github.com/Nkluge-correa/Tucano/tree/main/evaluations/README.md). To learn more about our evaluation harness selection, [read our preprint](https://arxiv.org/abs/2411.07854).

|                 | Average | Calame-PT | Lambada-PT | ARC-PT | HellaSwag-PT |
|-----------------|---------|-----------|------------|--------|--------------|
| Llama-3.2-3B    | 52      | 58.43     | 49.1       | 43.25  | 57.2         |
| Granite-3.0-2b  | 51.63   | 56.36     | 47.55      | 42.56  | 60.05        |
| **Tucano-2b4**  | 43.58   | 59.06     | 37.67      | 30.43  | 47.17        |
| Llama-3.2-1B    | 42.95   | 51.83     | 41.02      | 33.5   | 45.44        |
| **Tucano-1b1**  | 41.55   | 58.24     | 34.7       | 30.43  | 42.84        |
| Gemma-2b        | 40.38   | 51.16     | 39.88      | 37.95  | 32.53        |
| Bloom-1b7       | 40.37   | 55.64     | 31.98      | 30.34  | 43.52        |
| **Tucano-630m** | 39.5    | 56.55     | 33.13      | 28.89  | 39.41        |
| Gemma-2-2b      | 39.21   | 56.7      | 47.1       | 24.19  | 28.85        |
| Bloom-1b1       | 38.18   | 52.94     | 30.22      | 29.83  | 39.74        |
| GlórIA-1b3      | 36.05   | 52.79     | 27.71      | 26.67  | 37.04        |
| **Tucano-160m** | 35.14   | 52.31     | 28.16      | 27.01  | 33.07        |
| Xglm-564m       | 34.55   | 50.58     | 27.42      | 25.56  | 34.64        |
| Bloom-560m      | 34.32   | 49.95     | 25.44      | 24.74  | 37.15        |
| TTL-460m        | 33.78   | 49.42     | 23.29      | 29.4   | 33           |
| mGPT-1b3        | 31.81   | 47.14     | 29.92      | 23.81  | 26.37        |
| TTL-160m        | 30.78   | 46.72     | 20.98      | 26.15  | 29.29        |
| Lola-v1         | 30.19   | 26.4      | 18.32      | 30.42  | 45.61        |
| GPorTuguese     | 28.92   | 40.61     | 22.98      | 22.48  | 29.62        |

## Cite as 🤗

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

## Aknowlegments

We gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing \& Analytics Lab.

## License

Tucano is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](../../LICENSE) file.
