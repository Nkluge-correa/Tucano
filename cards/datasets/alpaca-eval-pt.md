---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: output
    dtype: string
  - name: generator
    dtype: string
  - name: dataset
    dtype: string
  splits:
  - name: eval
    num_bytes: 824573
    num_examples: 805
  download_size: 467651
  dataset_size: 824573
configs:
- config_name: default
  data_files:
  - split: eval
    path: data/eval-*
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- pt
tags:
- evaluation
pretty_name: Alpaca-Eval-PT
size_categories:
- n<1K
---

# Alpaca-Eval-PT

- **Repository:** [TucanoBR/alpaca-eval-pt](https://huggingface.co/datasets/TucanoBR/alpaca-eval-pt)
- **Paper:** Taori et al. [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

## Dataset Summary

This dataset contains 805 translated samples (Portuguese) from the Alpaca dataset.

This dataset is used with the [`AlpacaEval`](https://github.com/tatsu-lab/alpaca_eval) library, an automatic evaluator for instruction-following language models.

## Languages

Portuguese

## Licensing

License: [cc-by-nc-4.0](https://www.creativecommons.org/licenses/by-nc/4.0/deed.en)

## Citation

```bibtex
@article{taori2023alpaca,
  title={Alpaca: A strong, replicable instruction-following model},
  author={Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and Dubois, Yann and Li, Xuechen and Guestrin, Carlos and Liang, Percy and Hashimoto, Tatsunori B},
  journal={Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html},
  volume={3},
  number={6},
  pages={7},
  year={2023}
}

@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  month = {5},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}
```
