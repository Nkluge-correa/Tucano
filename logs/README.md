# Logs and Plots

This folder contains all logs related to the training and evaluation of the Tucano series.

- [`/alpaca-evals`](./alpaca-evals) contais the [results](./alpaca-evals/leaderboard.csv), [annotations](./alpaca-evals/annotations-Tucano-2b4-Instruct.json), and [generations](./alpaca-evals/output-Tucano-2b4-Instruct.json) used for our [`alpaca-eval-pt`](https://huggingface.co/datasets/TucanoBR/alpaca-eval-pt) benchmark.
- [`/emissions`](./emissions) contains all recorded [emissions](./emissions/emissions-xl.csv) from our training runs.
- [`/evals`](./evals) contains the evaluation results from all our other benchmarks (CALAME-PT, LAMBADA-PT, ENEM, BLUEX, OAB Exams, ASSIN2 RTE, ASSIN2 STS, FAQUAD NLI, HateBR, PT Hate Speech, TweetSentBR, ARC-Challenge, HellaSwag, TruthfulQA) for all our models.
- [`/training-logs`](./training-logs) contains the training and validation logs from our training runs.

The plots associated with our study are manipulated and created using the following [notebook](./logs-and-plots.ipynb).

## Tokenizer Test

In the notebook, you can also find the tokenizer test described in our [paper](https://arxiv.org/abs/xxxx.xxxxx). The text sample used to access the compression capabilities of the tokenizers can be found in the [tokenizer-test-set.txt](./evals/tokenizer-test-set.txt) file. Below, you can find a list of the tokenizers we ran evaluation:

- [`TucanoBR/Tucano-1b1`](https://huggingface.co/TucanoBR/Tucano-1b1)
- [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [`pablocosta/bertabaporu-base-uncased`](https://huggingface.co/pablocosta/bertabaporu-base-uncased)
- [`sagui-nlp/debertinha-ptbr-xsmall`](https://huggingface.co/sagui-nlp/debertinha-ptbr-xsmall)
- [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese)
- [`NOVA-vision-language/GlorIA-1.3B`](https://huggingface.co/NOVA-vision-language/GlorIA-1.3B)
- [`PORTULAN/gervasio-7b-portuguese-ptbr-decoder`](https://huggingface.co/PORTULAN/gervasio-7b-portuguese-ptbr-decoder)
- [`PORTULAN/gervasio-7b-portuguese-ptpt-decoder`](https://huggingface.co/PORTULAN/gervasio-7b-portuguese-ptpt-decoder)
- [`PORTULAN/albertina-100m-portuguese-ptbr-encoder`](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder)
- [`PORTULAN/albertina-100m-portuguese-ptpt-encoder`](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptpt-encoder)
- [`unicamp-dl/ptt5-base-portuguese-vocab`](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab)
- [`eduagarcia/RoBERTaCrawlPT-base`](https://huggingface.co/eduagarcia/RoBERTaCrawlPT-base)
- [`eduagarcia/RoBERTaLexPT-base`](https://huggingface.co/eduagarcia/RoBERTaLexPT-base)
- [`raquelsilveira/legalbertpt_sc`](https://huggingface.co/raquelsilveira/legalbertpt_sc)
- [`22h/open-cabrita3b`](https://huggingface.co/22h/open-cabrita3b)
- [`maritaca-ai/sabia-7b`](https://huggingface.co/maritaca-ai/sabia-7b)
- [`maritaca-ai/sabia-2-tokenizer-medium`](https://huggingface.co/maritaca-ai/sabia-2-tokenizer-medium)
- [`recogna-nlp/bode-7b-alpaca-pt-br-no-peft`](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br-no-peft)
- [`ai-forever/mGPT`](https://huggingface.co/ai-forever/mGPT)
- [`google-bert/bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [`bigscience/bloom`](https://huggingface.co/bigscience/bloom)
- [`facebook/xglm-564M`](https://huggingface.co/facebook/xglm-564M)
- [`DAMO-NLP-MT/polylm-1.7b`](https://huggingface.co/DAMO-NLP-MT/polylm-1.7b)
- [`botbot-ai/CabraLlama3-8b`](https://huggingface.co/botbot-ai/CabraLlama3-8b)
- [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [`google/mt5-base`](https://huggingface.co/google/mt5-base)
- [`unicamp-dl/ptt5-large-portuguese-vocab`](https://huggingface.co/unicamp-dl/ptt5-large-portuguese-vocab)
