---
license: apache-2.0
datasets:
- TucanoBR/GigaVerbo-Text-Filter
language:
- pt
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- text-quality
- portuguese
widget:
- text: "Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente."
  example_title: Sample 1
- text: "12 de março de 2021 | São Paulo 8 de agosto de 1999 | Porto Alegre 25 de dezembro de 2022 | Rio de Janeiro 17 de julho de 1985 | Lisboa 4 de outubro de 2010 | Belo Horizonte 23 de setembro de 1978 | Paris 14 de fevereiro de 2003 | Nova Iorque 19 de junho de 1994 | Brasília 5 de novembro de 2009 | Curitiba 30 de abril de 2015 | Buenos Aires"
  example_title: Sample 2
---
# BERTimbau-base-text-filter

BERTimbau-base-text-filter is a [BERT](https://huggingface.co/neuralmind/bert-base-portuguese-cased) model that can be used to score the quality of a given Portuguese text string. This model was trained on the [GigaVerbo-Text-Filter](https://huggingface.co/datasets/TucanoBR/GigaVerbo-Text-Filter) dataset.

## Details

- **Size:** 109,038,209 parameters
- **Dataset:** [GigaVerbo-Text-Filter](https://huggingface.co/datasets/TucanoBR/GigaVerbo-Text-Filter)
- **Language:** Portuguese
- **Number of Training Epochs:** 3
- **Batch size:** 128
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 4e-5

This repository has the [source code](https://github.com/Nkluge-correa/Tucano) used to train this model.

## Usage

Here's an example of how to use the BERTimbau-base-text-filter:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("TucanoBR/BERTimbau-base-text-filter")
model = AutoModelForSequenceClassification.from_pretrained("TucanoBR/BERTimbau-base-text-filter")
model.to(device)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
result = classifier("Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente.")
```

## Cite as 🤗

```latex
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

BERTimbau-base-text-filter is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](../../LICENSE) file.
