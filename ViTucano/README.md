<div align="center">
  
# ViTucano: A Portuguese Vision Assitant

<a href="https://huggingface.co/TucanoBR" target="_blank">
    <img src="https://img.shields.io/badge/HF%20Models-ViTucano-FFDF00.svg" alt="HF Link">
</a>
<a href="https://github.com/Nkluge-correa/Tucano/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache-002776.svg" alt="License: Apache 2.0">
</a>

</div>
<p align="center">
        <img src="./img/ViTucano-logo.png" alt="Uma ilustração de um tucano usando um elegante terno. O tucano está olhando para o lado, o que mostra o monóculo em seu olho direito." height="400">
</p>

We introduce **ViTucano**, our first attempt at creating a vision assistant natively pretrained in Portuguese. **ViTucano** is built on top of the Tucano series using the using the [TinyLLaVA Factory](https://arxiv.org/abs/2405.11788). ViTucano integrates visual understanding with linguistic capabilities, creating a tool for multimodal tasks. All resources from this development are openly available on GitHub and Hugging Face.

- [ViTucano-1b5-v1](https://huggingface.co/TucanoBR/ViTucano-1b5-v1)
- [ViTucano-2b8-v1](https://huggingface.co/TucanoBR/ViTucano-2b8-v1)

## Reproducing

To reproduce ViTucano, you first need to clone our [fork from the original TinyLLaVA Factory](https://github.com/Nkluge-correa/TinyLLaVA_Factory) and follow these installation instructions:

```bash
git clone https://github.com/Nkluge-correa/TinyLLaVA_Factory
cd TinyLLaVA_Factory
pip3 install -e .
pip3 install wheel
pip3 install flash-attn --no-build-isolation
```

Reproducing a ViTucano model, like all similar LLaVA models, requires two distinct steps: `feature-alignment` and `visual-instruction-tuning`.

Given that our study was performed in the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin), which uses SLURM for job scheduling, all major scripts were launched via bash scripts. All bash scripts used carry the corresponding names of their corresponding Python scripts and can be found in the [`./scripts`](./scripts/README.md) folder. All logs (e.g., training logs) related to training the ViTucano models can be found in the [`/logs`](./logs/README.md) folder.

### Feature Alignment Corpus (ViTucano-Pretrain)

To train the projector (i.e., feature alignment), we used the [ViTucano-Pretrain](https://huggingface.co/datasets/TucanoBR/ViTucano-Pretrain) dataset. This dataset is a translation of the original [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), obtained via Google's translation API. LLaVA Visual Instruct Pretrain LCS-558K is a subset of the LAION/CC/SBU dataset, filtered with a more balanced concept coverage distribution.

| Hyperparameters | Global Batch Size | Learning rate | Epochs | Weight decay |
|-----------------|-------------------|---------------|--------|--------------|
|                 | 256               | 1e-3          | 1      |  0           |

### Visual Instruction Tuning Corpus (ViTucano-SFT)

For visual instruction tuning, we used samples from the original [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) dataset. More specifically, only the samples from the `coco` and `gqa` partitions are needed. These samples were then translated into Portuguese using Google's translation API. The original dataset ([LLaVA Visual Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)) was created by prompting GPT-4-0314 API. We also added samples (i.e., the entire train portion) from the [COCO Captions Portuguese Translation](https://huggingface.co/datasets/laicsiifes/coco-captions-pt-br). This concatenation is available in [TucanoBR/ViTucano-SFT](https://huggingface.co/datasets/TucanoBR/ViTucano-SFT).

| Hyperparameters | Global Batch Size | Learning rate | Epochs | Weight decay |
|-----------------|-------------------|---------------|--------|--------------|
|                 | 128               | 2e-5          | 4      | 0            |

## Basic usage

⚠️Using ViTucano models through the `transformers` library requires executing remote code (`trust_remote_code=True`). The executed files are [`configuration.py`](./configuration.py) and [`modeling_tinyllava_tucano.py`](./modeling_tinyllava_tucano.py), both available in this repository.⚠️

<details>
<summary>Run inference using <code>tinyllava</code></summary>

```python
from tinyllava.eval.run_tiny_llava import eval_model

model_path = "TucanoBR/ViTucano-2b8-v1"
prompt = "Quais os principais elementos dessa imagem?"
image_file = "https://raw.githubusercontent.com/Nkluge-correa/TinyLLaVA_Factory/refs/heads/main/assets/sample.jpg"
conv_mode = "llama"

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

<details>
<summary>Run inference using <code>transformers</code></summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "TucanoBR/ViTucano-2b8-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  #torch_dtype=torch.bfloat16, # for optimized inference  🚀
  #attn_implementation="flash_attention_2" # for optimized inference  🚀
  trust_remote_code=True)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "Quais os principais elementos dessa imagem?"
image_file="https://raw.githubusercontent.com/Nkluge-correa/TinyLLaVA_Factory/refs/heads/main/assets/sample.jpg"
output_text, _ = model.chat(prompt=prompt, image=image_file, tokenizer=tokenizer)

print(output_text)
```
</details>

## Intended Uses

The primary intended use of the ViTucano models is to serve as foundations for research and development involving native Portuguese foundation models. You may also fine-tune and adapt ViTucano models for deployment if your use follows the Apache 2.0 license. If you decide to use the ViTucano models as a basis for your fine-tuned model, please conduct your own risk and bias assessment.

## Out-of-scope Use

- ViTucano models are **not intended for deployment**. They are not an out-of-the-box product and should not be used for human-facing interactions.

- ViTucano models are for **the Portuguese language only** and are unsuitable for image-to-text generation tasks in other languages.

- ViTucano models have **not been fine-tuned** for any specific downstream task.

## Limitations

Like almost all other multimodal language models trained on large datasets scraped from the web, the ViTucano models show behavior that does not make them an out-of-the-box solution to many real-world applications, especially those requiring factual, reliable, and nontoxic text generation. ViTucano models are all subject to the following:

- **Hallucinations:** ViTucano models may generate misleading or entirely false information when interpreting or describing visual inputs, leading to hallucinations that could be mistaken as accurate observations or factual statements.

- **Biases and Toxicity:** ViTucano models inherit social and historical stereotypes in the training data. These biases can manifest in harmful, offensive, or misleading descriptions or analyses of visual or textual content.

- **Unreliable Visual Interpretations:** ViTucano models may produce inaccurate interpretations of visual elements, including objects, scenes, or text within images. Such outputs should not be considered reliable without human verification.

- **Multimodal Language Limitations:** While ViTucano models are optimized for Portuguese, handling multilingual visual and textual contexts may lead to errors, misinterpretations, or inadequate responses, especially with non-Portuguese content.

- **Repetition and Irrelevant Details:** ViTucano models can exhibit repetitive response patterns or generate verbose descriptions unrelated to the given visual or textual input, particularly under specific hyperparameter configurations.

Hence, even though our models are released with a permissive license, we urge users to perform their risk analysis before using them for real-world applications.

## Cite as 🤗

### ViTucano

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

### Tucano

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

### TinyLLaVA Factory

```bibtex
@article{jia2024tinyllava,
  title={TinyLLaVA Factory: A Modularized Codebase for Small-scale Large Multimodal Models},
  author={Jia, Junlong and Hu, Ying and Weng, Xi and Shi, Yiming and Li, Miao and Zhang, Xingjian and Zhou, Baichuan and Liu, Ziyu and Luo, Jie and Huang, Lei and Wu, Ji},
  journal={arXiv preprint arXiv:2405.11788},
  year={2024}
}
```

### LLaVA

```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

## Aknowlegments

We gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing \& Analytics Lab.

## License

ViTucano is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](./LICENSE) file.
