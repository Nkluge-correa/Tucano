---
language:
- pt
license: apache-2.0
library_name: transformers
tags:
- text-generation-inference
datasets:
- nicholasKluge/instruct-aira-dataset-v3
- cnmoro/GPT4-500k-Augmented-PTBR-Clean
- rhaymison/orca-math-portuguese-64k
- nicholasKluge/reward-aira-dataset
metrics:
- perplexity
pipeline_tag: text-generation
widget:
- text: "<instruction>Cite algumas bandas de rock brasileiras famosas.</instruction>"
  example_title: Exemplo
- text: "<instruction>Invente uma história sobre um encanador com poderes mágicos.</instruction>"
  example_title: Exemplo
- text: "<instruction>Qual cidade é a capital do estado do Rio Grande do Sul?</instruction>"
  example_title: Exemplo
- text: "<instruction>Diga o nome de uma maravilha culinária característica da cosinha Portuguesa?</instruction>"
  example_title: Exemplo
inference:
  parameters:
    repetition_penalty: 1.2
    temperature: 0.2
    top_k: 20
    top_p: 0.2
    max_new_tokens: 150
co2_eq_emissions:
  emissions: 21890
  source: CodeCarbon
  training_type: pre-training
  geographical_location: Germany
  hardware_used: NVIDIA A100-SXM4-80GB
model-index:
- name: Tucano-1b1-Instruct
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
      value: 56.55
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
      value: 35.53
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
      value: 21.06
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
      value: 26.01
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
      value: 26.47
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
      value: 67.78
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
      value: 8.88
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
      value: 31.28
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
      value: 41.23
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
      value: 22.03
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
      value: 30.77
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
      value: 43.50
      name: normalized accuracy
    source:
      url: https://github.com/nlp-uoregon/mlmm-evaluation
      name: Evaluation Framework for Multilingual Large Language Models
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: TruthfulQA (PT)
      type: truthfulqa_pt
      args:
        num_few_shot: 0
    metrics:
    - type: mc2
      value: 41.14
      name: bleurt
    source:
      url: https://github.com/nlp-uoregon/mlmm-evaluation
      name: Evaluation Framework for Multilingual Large Language Models
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Alpaca-Eval (PT)
      type: alpaca_eval_pt
      args:
        num_few_shot: 0
    metrics:
    - type: lc_winrate
      value: 8.80
      name: length controlled winrate
    source:
      url: https://github.com/tatsu-lab/alpaca_eval
      name: AlpacaEval
base_model:
- TucanoBR/Tucano-1b1
---
# Tucano-1b1-Instruct

<img src="./logo.png" alt="An illustration of a Tucano bird showing vibrant colors like yellow, orange, blue, green, and black." height="200">

## Model Summary

Tucano-1b1-Instruct is a fine-tuned version of [Tucano-1b1](https://huggingface.co/TucanoBR/Tucano-1b1). **[Tucano](https://huggingface.co/TucanoBR)** is a series of decoder-transformers based on the Llama 2 architecture, pretrained natively in Portuguese. All Tucano models were trained on **[GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo)**, a concatenation of deduplicated Portuguese text corpora amounting to 200 billion tokens.

The fine-tuning process was divided into two stages:

- Supervised fine-tuning (SFT) using the [TucanoBR/Tucano-SFT](https://huggingface.co/datasets/TucanoBR/Tucano-SFT), a concatenation of three different instruction tuning datasets ([`cnmoro/GPT4-500k-Augmented-PTBR-Clean`](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean), [`rhaymison/orca-math-portuguese-64k`](https://huggingface.co/datasets/rhaymison/orca-math-portuguese-64k), [`nicholasKluge/instruct-aira-dataset-v3`](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v3)).
- Direct Preference Optimization (DPO) using the [nicholasKluge/reward-aira-dataset](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset).

Read our preprint [here](https://arxiv.org/abs/xxxx.xxxxx).

## Details

- **Architecture:** a Transformer-based model pre-trained via causal language modeling
- **Size:** 1,100,048,384 parameters
- **Context length:** 2048 tokens
- **Dataset:**
  - [cnmoro/GPT4-500k-Augmented-PTBR-Clean](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean)
  - [rhaymison/orca-math-portuguese-64k](https://huggingface.co/datasets/rhaymison/orca-math-portuguese-64k)
  - [nicholasKluge/instruct-aira-dataset-v3](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v3)
  - [nicholasKluge/reward-aira-dataset](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset)
- **Language:** Portuguese
- **Training time**: ~ 12 hours
- **Emissions:** 22 KgCO2 (Germany)
- **Total energy consumption:** 58 kWh

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

generator = pipeline("text-generation", model="TucanoBR/Tucano-1b1-Instruct")

completions  = generator("<instruction>Qual cidade é a capital do estado do Rio Grande do Sul?</instruction>", num_return_sequences=2, max_new_tokens=100)

for comp in completions:
  print(f"🤖 {comp['generated_text']}")
```

Using the `AutoTokenizer` and `AutoModelForCausalLM`:

```python
from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the model and tokenizer
model_id = "TucanoBR/Tucano-1b1-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Specify the generation parameters as you like
generation_config = GenerationConfig(
    **{
    "do_sample": True,
    "max_new_tokens": 2048,
    "renormalize_logits": True,
    "repetition_penalty": 1.2,
    "temperature": 0.3,
    "top_k": 30,
    "top_p": 0.3,
    "use_cache": True, 
  }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = TextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, device=device)

# Generate text
prompt = "<instruction>Qual cidade é a capital do estado do Rio Grande do Sul?</instruction>"
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

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis on them if they intend to use them for real-world applications. We also have humans moderating the outputs of these models in applications where they will interact with an audience, guaranteeing users are always aware they are interacting with a language model.

## Evaluations

To evaluate the `Instruct` versions of our models, we used [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) 2.0 with length-controlled win rates, a fast and relatively cheap evaluation method that is highly correlated with human preferences and evaluations of pairwise comparisons. To learn more about our evaluation read [our documentation](https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md).

|                         | Avg. Length | Wins | Base Wins | Total Matches | Length-Controlled Win Rate (%) | LC Std. Error |
|-------------------------|-------------|------|-----------|---------------|--------------------------------|---------------|
| Llama-3.2-3B-Instruct   | 1609        | 257  | 548       | 805           | 21.06                          | 0.075         |
| **Tucano-2b4-Instruct** | 1843        | 151  | 654       | 805           | 13.00                          | 0.071         |
| **Tucano-1b1-Instruct** | 1667        | 124  | 681       | 805           | 8.80                           | 0.083         |
| Llama-3.2-1B-Instruct   | 1429        | 99   | 706       | 805           | 7.15                           | 0.057         |
| TeenyTinyLlama-460m-Chat| 1333        | 28   | 777       | 805           | 2.84                           | 0.059         |
| Sabiá-7b                | 5011        | 1    | 804       | 805           | 0.076                          | 0.0043        |
| Gervásio-7b             | 5740        | 1    | 804       | 805           | 0.026                          | 0.0016        |

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
