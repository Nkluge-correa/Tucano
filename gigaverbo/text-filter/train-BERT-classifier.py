import evaluate
import argparse
import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset, Dataset

def load_datasets(dataset_path, hub_token):
    """Load the dataset from the Hugging Face Hub"""
    dataset = load_dataset(
        dataset_path,
        split="train",
        token=hub_token
    )
    return dataset

def preprocess_data(dataset):
    """Preprocess the dataset"""
    high = dataset.filter(lambda x: x['score'] >= 0.8)
    low = dataset.filter(lambda x: x['score'] <= 0.6)
    high = high.map(lambda x: {**x, 'labels': 1})
    low = low.map(lambda x: {**x, 'labels': 0})
    combined = Dataset.from_dict({key: high[key] + low[key] for key in high.features.keys()})
    combined = combined.remove_columns(['dataset', 'embedding', 'score'])
    ds = combined.train_test_split(test_size=0.1)
    return ds

def main(args):
    dataset = load_datasets(args.dataset_path, args.hub_token)
    ds = preprocess_data(dataset)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "LOW", 1: "HIGH"},
        label2id={"LOW": 0, "HIGH": 1}
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    dataset_tokenized = ds.map(preprocess_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.evaluate:
        from sklearn.metrics import classification_report, confusion_matrix
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import TextClassificationPipeline
        import plotly.express as px
        import torch
        import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(args.hub_model_id, token=args.hub_token)
        model = AutoModelForSequenceClassification.from_pretrained(args.hub_model_id, token=args.hub_token)
        model.to(device)

        classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

        test_texts = ds['test']['text']
        encoded_inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}

        batch_size = args.eval_batch_size
        predictions = []

        for i in tqdm.tqdm(range(0, len(test_texts), batch_size)):
            batch = {key: val[i:i+batch_size] for key, val in encoded_inputs.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

        label_map = {0: 0, 1: 1}  # Assuming the model's labels are [0: 'low', 1: 'high']
        mapped_predictions = [label_map[pred] for pred in predictions]

        target_names = ['low', 'high']
        print(classification_report(ds['test']['labels'], mapped_predictions, target_names=target_names))

        matrix = confusion_matrix(ds['test']['labels'], mapped_predictions)

        fig = px.imshow(matrix,
                        labels=dict(x="Predicted", y="True label"),
                        x=target_names,
                        y=target_names,
                        text_auto=True)

        fig.update_xaxes(side='top')
        fig.update_layout(template='plotly_dark',
                        title='Confusion Matrix',
                        coloraxis_showscale=False,
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for text classification")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset in the Hugging Face Hub")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save the model checkpoints")
    parser.add_argument('--learning-rate', type=float, default=4e-5, help="Learning rate for training")
    parser.add_argument('--train-batch-size', type=int, default=32, help="Training batch size")
    parser.add_argument('--eval-batch-size', type=int, default=32, help="Evaluation batch size")
    parser.add_argument('--num-train-epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument('--hub-token', type=str, required=True, help="Hugging Face Hub token for authentication")
    parser.add_argument('--hub-model-id', type=str, required=True, help="Model ID to push to Hugging Face Hub")
    parser.add_argument('--evaluate', type=bool, default=False, help="Evaluate the model after training")
    args = parser.parse_args()
    main(args)

# python3 train-BERT-classifier.py \
#--dataset-path "TucanoBR/GigaVerbo-Text-Filter" \
#--model-name "neuralmind/bert-base-portuguese-cased" \
#--output-dir "path/to/checkpoints" \
#--learning-rate 4e-5 \
#--train-batch-size 32 \
#--eval-batch-size 32 \
#--num-train-epochs 3 \
#--weight-decay 0.01 \
#--hub-token None \
#--hub-model-id "userName/modelName" \
#--evaluate True
#
# How to use the trained model:
#
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from transformers import TextClassificationPipeline
#import torch
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#tokenizer = AutoTokenizer.from_pretrained("TucanoBR/BERTimbau-base-text-filter")
#model = AutoModelForSequenceClassification.from_pretrained("TucanoBR/BERTimbau-base-text-filter")
#model.to(device)
#
#classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
#result = classifier("Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente.")
