---
dataset_info:
  features:
  - name: sentence
    dtype: string
  - name: last_word
    dtype: string
  splits:
  - name: train
    num_bytes: 1844684
    num_examples: 5153
  download_size: 1241703
  dataset_size: 1844684
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: mit
task_categories:
- text-generation
language:
- pt
pretty_name: LAMBADA-PT
size_categories:
- 1K<n<10K
---

# LAMBADA-PT

- **Repository:** [TucanoBR/lambada-pt](https://huggingface.co/datasets/TucanoBR/lambada-pt)
- **Paper:** Radford et al. [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

## Dataset Summary

This dataset is a translated version (Portuguese) of the LAMBADA test split as pre-processed by OpenAI.

LAMBADA is used to evaluate the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative texts sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole text, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse.

## Languages

Portuguese

## Licensing

License: [Modified MIT](https://github.com/openai/gpt-2/blob/master/LICENSE)

## Citation

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```
