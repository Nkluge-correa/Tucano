---
license: apache-2.0
datasets:
- TucanoBR/GigaVerbo-Text-Filter
language:
- pt
metrics:
- mse
library_name: xgboost
tags:
- text-quality
- portuguese
---
# XGBRegressor-text-filter

XGBRegressor-text-filter is a text-quality filter built on top of the [`xgboost`](https://xgboost.readthedocs.io/en/stable/) library. It uses the embeddings generated by [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) as a feature vector.

This repository has the [source code](https://github.com/Nkluge-correa/Tucano) used to train this model.

## Usage

Here's an example of how to use the XGBRegressor-text-filter:

```python
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBRegressor
import torch.nn.functional as F
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
embedding_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
device = ("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

bst_r = XGBRegressor({'device': device})
bst_r.load_model('/path/to/XGBRegressor-text-classifier.json')

def score_text(text, model):

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = embedding_model(**encoded_input)

    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    embedding = F.normalize(sentence_embedding, p=2, dim=1).numpy()
    score = model.predict(embedding)[0]

    return score

score_text("Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente.", bst_r)
```

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

XGBRegressor-text-filter is licensed under the Apache License, Version 2.0. For more details, see the [LICENSE](../../LICENSE) file.