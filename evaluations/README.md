# Evaluations

During training, we saved checkpoints for each model between an interval of $\approx$ 10.5 billion tokens. For every checkpoint, we ran the following harness of evaluation benchmarks:

| **Benchmark**     | **n-shot** | **Origin**   | **Type**              | **Metric**        |
|-------------------|------------|--------------|-----------------------|-------------------|
| ENEM              | 3-shot     | Native       | Q&A                   | `acc`             |
| BLUEX             | 3-shot     | Native       | Q&A                   | `acc`             |
| OAB Exams         | 3-shot     | Native       | Q&A                   | `acc`             |
| ASSIN2 RTE        | 15-shot    | Native       | Entailment            | `f1 macro`        |
| ASSIN2 STS        | 10-shot    | Native       | Similarity            | `pearson`         |
| FAQUAD NLI        | 15-shot    | Native       | Entailment            | `f1 macro`        |
| HateBR            | 25-shot    | Native       | Classification        | `f1 macro`        |
| PT Hate Speech    | 25-shot    | Native       | Classification        | `f1 macro`        |
| TweetSentBR       | 25-shot    | Native       | Classification        | `f1 macro`        |
| CALAME-PT         | 0-shot     | Native       | Next Word Prediction  | `acc`             |
| ARC-Challenge     | 25-shot    | Translated   | Q&A                   | `acc norm`        |
| HellaSwag         | 10-shot    | Translated   | Q&A                   | `acc norm`        |
| TruthfulQA        | 0-shot     | Translated   | Q&A                   | `bleurt`          |
| LAMBADA           | 0-shot     | Translated   | Next Word Prediction  | `acc`             |

For CALAME-PT and LAMBADA-PT, the scripts used to run evaluations are:

- [`lm-calame-pt-eval.py`](./lm-calame-pt-eval.py).
- [`lm-lambada-pt-eval.py`](./lm-lambada-pt-eval.py).

Other evaluations are performed by using either the [NLP-UOregon](https://github.com/nlp-uoregon) multilingual harness or Eduardo Garcia's [Portuguese evaluation harness](https://github.com/eduagarcia/lm-evaluation-harness-pt), both built on top [EleutherAI](https://www.eleuther.ai/) LM Evaluation Harness. All scripts used to run these evaluations are found in the [`/scripts`](../scripts/) folder:

- [`lm-mlmm-evaluation.sh`](../scripts/lm-mlmm-evaluation.sh).
- [`lm-pt-evaluation.sh`](../scripts/lm-pt-evaluation.sh).

The table below brings a comparison of the above-cited benchmark evaluations on several Portuguese and multilingual language models:

|                 | Average | Calame-PT | Lambada-PT | Enem  | Bluex | OAB Exams | Assin2 RTE | Assin2 STS | FAQUAD-NLI | HateBR | HateSpeech-PT | TweetBR | ARC-PT | HellaSwag-PT | TruthfulQA-PT |
|-----------------|---------|-----------|------------|-------|-------|-----------|------------|------------|------------|--------|---------------|---------|--------|--------------|---------------|
| Granite-3.0-2b  | 54.9    | 56.36     | 47.55      | 54.51 | 45.2  | 40.46     | 83.72      | 60.46      | 43.97      | 55.81  | 68.12         | 67.6    | 42.56  | 60.05        | 42.23         |
| Llama-3.2-3B    | 52.38   | 58.43     | 49.1       | 53.04 | 50.35 | 39.45     | 83.64      | 33.19      | 43.97      | 74.58  | 41.99         | 61.43   | 43.25  | 57.2         | 43.64         |
| Gemma-2-2b      | 48.47   | 56.7      | 47.1       | 49.34 | 40.47 | 35.54     | 76.17      | 36.04      | 44.71      | 65.08  | 58.31         | 66.75   | 24.19  | 28.85        | 49.38         |
| Gemma-2b        | 41.66   | 51.16     | 39.88      | 26.31 | 28.79 | 28.29     | 64.8       | 20.69      | 44.07      | 77.69  | 36.05         | 53.07   | 37.95  | 32.53        | 41.96         |
| Llama-3.2-1B    | 38.67   | 51.83     | 41.02      | 23.37 | 24.2  | 25.88     | 50.77      | 19.48      | 43.97      | 59.43  | 38.57         | 42.34   | 33.5   | 45.44        | 41.63         |
| **Tucano-2b4**  | 36.75   | 59.06     | 37.67      | 20.5  | 23.23 | 25.47     | 56.27      | 1.93       | 43.97      | 29.49  | 41.98         | 58      | 30.43  | 47.17        | 39.3          |
| **Tucano-1b1**  | 36.45   | 58.24     | 34.7       | 21.41 | 23.37 | 25.97     | 60.82      | 24.63      | 43.97      | 29     | 41.19         | 32.18   | 30.43  | 42.84        | 41.59         |
| **Tucano-630m** | 34.16   | 56.55     | 33.13      | 19.17 | 24.76 | 25.28     | 57.79      | 1.99       | 43.97      | 53.73  | 30.01         | 20.73   | 28.89  | 39.41        | 42.76         |
| Bloom-1b1       | 33.04   | 52.94     | 30.22      | 19.87 | 22.11 | 24.74     | 54.32      | 14.64      | 43.97      | 38.45  | 35.64         | 15.07   | 29.83  | 39.74        | 41.04         |
| Bloom-1b7       | 32.88   | 55.64     | 31.98      | 18.96 | 21.42 | 23.05     | 53.6       | 4.81       | 43.97      | 34.89  | 41.23         | 15.07   | 30.34  | 43.52        | 41.86         |
| Xglm-564m       | 31.42   | 50.58     | 27.42      | 19.03 | 19.75 | 23.55     | 49.9       | 23.35      | 43.97      | 33.99  | 24.9          | 20.73   | 25.56  | 34.64        | 42.53         |
| TTL-460m        | 31.14   | 49.42     | 23.29      | 20.15 | 25.73 | 27.02     | 53.61      | 13         | 46.41      | 33.59  | 22.99         | 17.28   | 29.4   | 33           | 41.1          |
| TTL-160m        | 29.86   | 46.72     | 20.98      | 19.24 | 23.09 | 22.37     | 53.97      | 0.24       | 43.97      | 36.92  | 42.63         | 11.39   | 26.15  | 29.29        | 41.12         |
| **Tucano-160m** | 29.52   | 52.31     | 28.16      | 19.03 | 22.11 | 25.1      | 33.51      | 11.02      | 43.97      | 36.56  | 22.99         | 16.86   | 27.01  | 33.07        | 41.53         |
| Bloom-560m      | 29.19   | 49.95     | 25.44      | 19.03 | 18.92 | 23.05     | 33.33      | 8.48       | 43.97      | 37.07  | 24.29         | 20.74   | 24.74  | 37.15        | 42.44         |
| GPorTuguese     | 25.14   | 40.61     | 22.98      | 19.31 | 21.42 | 3.14      | 33.59      | 3.44       | 43.97      | 33.33  | 22.99         | 13.62   | 22.48  | 29.62        | 41.44         |
| mGPT-1b3        | 18.1    | 47.14     | 29.92      | 16.66 | 10.43 | 8.56      | 0          | 0.58       | 0          | 10.79  | 28.12         | 11.36   | 23.81  | 26.37        | 39.62         |
| GlórIA-1b3      | 15.96   | 52.79     | 27.71      | 1.89  | 3.2   | 5.19      | 0          | 2.32       | 0.26       | 0.28   | 23.52         | 0.19    | 26.67  | 37.04        | 42.44         |
| Lola-v1         | 11.19   | 26.4      | 18.32      | 0     | 0     | 0         | 0          | 0          | 0          | 0      | 0             | 0.43    | 30.42  | 45.61        | 35.54         |

## MTEB

To test the [`LLM2Vec`](https://github.com/McGill-NLP/llm2vec) methodology proposed in "[LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](https://mcgill-nlp.github.io/llm2vec/)", we concatenated a Portuguese evaluation harness for embedding/text-similarity models. The script used to run this evaluation is:

- [`mteb-custom-eval.py`](./mteb-custom-eval.py).

Make sure to install llm2vec before running it:

```bash
pip install llm2vec -q
pip install llm2vec[evaluation] -q
```

All benchmarks used in our Portuguese MTEB evaluation are listed below:

- [MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages](https://arxiv.org/abs/2204.08582)
  - `mteb/amazon_massive_intent`
  - `mteb/amazon_massive_scenario`
  
- [MultiHate Classification Dataset for Hate Speech Detection](https://aclanthology.org/2022.woah-1.15/)
  - `mteb/multi-hatecheck`
  
- [SIB-200 Topic Classification Dataset](https://arxiv.org/abs/2309.07445)
  - `mteb/sib200`
  
- [Multilingual Tweet Sentiment Analysis Dataset](https://aclanthology.org/2022.lrec-1.27)
  - `mteb/tweet_sentiment_multilingual`
  
- [Hate Speech Portuguese Classification Dataset](https://aclanthology.org/W19-3510)
  - `hate-speech-portuguese/hate_speech_portuguese`
  
- [Mintaka: Multilingual Question-Answering Dataset](https://aclanthology.org/2022.coling-1.138)
  - `jinaai/mintakaqa`
  
- [Wikipedia Retrieval Multilingual Dataset](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries)
  - `ellamind/wikipedia-2023-11-retrieval-multilingual-queries`
  
- [ASSIN 2: Recognizing Textual Entailment and Semantic Textual Similarity Dataset](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)
  - `nilc-nlp/assin2` (for both RTE and STS tasks)
  
- [STS Benchmark Multilingual - STS Benchmark Dataset Translated](https://github.com/PhilipMay/stsb-multi-mt/)
  - `mteb/stsb_multi_mt`

## Alpaca-Eval-PT

To evaluate [Tucano-1b1-Instruct](https://huggingface.co/TucanoBR/Tucano-1b1-Instruct) and [Tucano-2b4-Instruct](https://huggingface.co/TucanoBR/Tucano-2b4-Instruct), we used [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) 2.0 with length-controlled win rates, a fast and relatively cheap evaluation method that is highly correlated with human preferences and evaluations of pairwise comparisons.

All `instruction` and `output` pais are generated via the [TucanoBR/alpaca-eval-pt](https://huggingface.co/datasets/TucanoBR/alpaca-eval-pt), which holds 805 completion samples generated by `text-davinci-003`. These also serve as references for the alpaca-eval procedure, i.e., all generated samples are compared with the `text-davinci-003` original generations. Win rates are calculated using [`alpaca_eval_cot_gpt4_turbo_fn`](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#evaluators) as an annotator.

To create the outputs of a model, use:

- [`alpaca-eval-pt.py`](./alpaca-eval-pt.py). You can find an example output file [here](../logs/alpaca-evals/output-Tucano-2b4-Instruct.json).

After you already have your output file (the reference file can be found [here](../logs/alpaca-evals/reference.json)), install `alpaca-eval`, set your OpenAI API key, and run the eval:

```bash
pip3 install alpaca-eval
export OPENAI_API_KEY="sk-..."

alpaca_eval --model_outputs="/logs/alpaca-evals/output-Tucano-2b4-Instruct.json" --reference_outputs="/logs/alpaca-evals/reference.json" --output_path="path/to/folder"
```

The table below provides comparisons for our evaluation:

|                          | Win Rate (%) | Std. Error | Avg. Length | Wins | Base Wins | Draws | Total Matches | Discrete Win Rate (%) | Length-Controlled Win Rate (%) | LC Std. Error |
|--------------------------|--------------|------------|-------------|------|-----------|-------|---------------|-----------------------|--------------------------------|---------------|
| Llama-3.2-3B-Instruct    | 31.81        | 1.46       | 1609        | 257  | 548       | 0     | 805           | 31.92                 | 21.06                          | 0.075         |
| **Tucano-2b4-Instruct**  | 18.90        | 1.23       | 1843        | 151  | 654       | 0     | 805           | 18.75                 | 13.00                          | 0.071         |
| **Tucano-1b1-Instruct**  | 15.41        | 1.16       | 1667        | 124  | 681       | 0     | 805           | 15.40                 | 8.80                           | 0.083         |
| Llama-3.2-1B-Instruct    | 13.00        | 1.05       | 1429        | 99   | 706       | 0     | 805           | 12.29                 | 7.15                           | 0.057         |
| TeenyTinyLlama-460m-Chat | 4.07         | 0.61       | 1333        | 28   | 777       | 0     | 805           | 3.47                  | 2.84                           | 0.059         |
| Sabiá-7b                 | 0.23         | 0.10       | 5011        | 1    | 804       | 0     | 805           | 0.12                  | 0.076                          | 0.0043        |
| Gervásio-7b              | 0.076        | 0.06       | 5740        | 1    | 804       | 0     | 805           | 0.12                  | 0.026                          | 0.0016        |
