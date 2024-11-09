# Pretraining Corpus (GigaVerbo Text-Filter)

This folder contains the scripts to train the filters used to parse GigaVerbo. All these filters are currently hosted on [Hugging Face](https://huggingface.co/datasets/TucanoBR). These are:

- Training of a BERT-based filter ([`train-BERT-classifier.py`](./train-BERT-classifier.py)).
- Training of the XGBClassifier ([`train-xgboost-classifier.py`](./train-xgboost-classifier.py)).
- Training of the XGBRegressor ([`train-xgboost-regressor.py`](./train-xgboost-regressor.py)).
- A script to run the BERT-based classifier on a text dataset ([`run-text-filter.py`](./run-text-filter.py)).

To learn more about GigaVerbo Text-Filter, read its [dataset card](../../cards/datasets/gigaverbo-text-filter.md).
