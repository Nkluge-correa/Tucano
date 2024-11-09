import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from xgboost import XGBClassifier
import plotly.express as px
import huggingface_hub

def load_datasets(dataset_path, huggingface_token):
    """Load the dataset from the Hugging Face Hub"""
    dataset = load_dataset(
        dataset_path,
        split="train",
        token=huggingface_token
    )

    dataset = dataset.to_pandas()
    return dataset

def preprocess_data(dataset):
    """Preprocess the data to create a binary classification dataset"""
    dataset['score'] = pd.to_numeric(dataset['score'])
    high = dataset[dataset['score'] >= 0.8].copy()
    low = dataset[dataset['score'] <= 0.6].copy()
    high['label'] = 1
    low['label'] = 0
    df = pd.concat([high, low]).drop(["dataset"], axis=1).reset_index(drop=True)
    embeddings_array = np.array(df['embedding'].tolist())
    embedding_df = pd.DataFrame(embeddings_array)
    embedding_df['label'] = df['label']
    return embedding_df

def split_data(embedding_df):
    """Split the data into train and test sets"""
    train, test = train_test_split(embedding_df, test_size=0.1, stratify=embedding_df['label'])
    classification_x_train = train.iloc[:, :-1]
    classification_y_train = train['label']
    classification_x_test = test.iloc[:, :-1]
    classification_y_test = test['label']
    return classification_x_train, classification_y_train, classification_x_test, classification_y_test

def train_model(classification_x_train, classification_y_train, classification_x_test, classification_y_test,
                learning_rate, max_depth, n_estimators, early_stopping_rounds, nthread):
    """Train the XGBoost classifier"""
    bst = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        early_stopping_rounds=early_stopping_rounds,
        booster='gbtree',
        objective='binary:logistic',
        nthread=nthread,
    )
    bst.fit(
        classification_x_train, classification_y_train,
        eval_set=[(classification_x_test, classification_y_test)],
        verbose=100
    )
    return bst

def save_and_upload_model(bst, output_path, huggingface_token, repo_id):
    """Save and upload the trained model to the Hugging Face Hub"""
    bst.save_model(output_path)
    api = huggingface_hub.HfApi(token=huggingface_token)
    huggingface_hub.create_repo(
        repo_id=repo_id,
        token=huggingface_token,
        repo_type="model",
        exist_ok=True,
        private=True)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_path.split("/")[-1],
        repo_id=repo_id
    )
    print("Model uploaded to the Hugging Face Hub")

def main(args):

    dataset = load_datasets(args.dataset_path, args.huggingface_token)
    embedding_df = preprocess_data(dataset)

    classification_x_train, classification_y_train, classification_x_test, classification_y_test = split_data(embedding_df)

    print("Train dataset: ", classification_x_train.shape)
    print("Train labels: ", classification_y_train.shape)
    print("Test dataset: ", classification_x_test.shape)
    print("Test labels: ", classification_y_test.shape)
    
    bst = train_model(classification_x_train, classification_y_train, classification_x_test, classification_y_test,
                      args.learning_rate, args.max_depth, args.n_estimators, args.early_stopping_rounds, args.nthread)
    preds = bst.predict(classification_x_test)
    target_names = ['low', 'high']
    print(classification_report(classification_y_test, preds, target_names=target_names))

    if args.plot_confusion_matrix:

        import plotly.express as px
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(classification_y_test, preds)
        fig = px.imshow(matrix,
                        labels=dict(x="Predicted", y="True label"),
                        x=target_names,
                        y=target_names,
                        text_auto=True
                        )
        fig.update_xaxes(side='top')
        fig.update_layout(template='plotly_dark',
                          title='Confusion Matrix',
                          coloraxis_showscale=False,
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)')
        fig.show()
    
    save_and_upload_model(bst, args.output_path, args.huggingface_token, args.repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload a text classifier")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset in the Hugging Face Hub")
    parser.add_argument('--learning-rate', type=float, default=0.1, help="Learning rate for the XGBoost classifier")
    parser.add_argument('--max-depth', type=int, default=10, help="Maximum depth of the trees in the XGBoost classifier")
    parser.add_argument('--n-estimators', type=int, default=100, help="Number of trees in the XGBoost classifier")
    parser.add_argument('--early-stopping-rounds', type=int, default=150, help="Early stopping rounds for the XGBoost classifier")
    parser.add_argument('--nthread', type=int, default=4, help="Number of threads to use for training")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--huggingface-token', type=str, required=True, help="Hugging Face token for authentication")
    parser.add_argument('--repo-id', type=str, required=True, help="Hugging Face repo id to upload the model")
    parser.add_argument('--plot-confusion-matrix', type=bool, default=True, help="Flag to plot the confusion matrix")
    args = parser.parse_args()
    main(args)

# python3 train-xgboost-classifier.py \
#--dataset-path "TucanoBR/GigaVerbo-Text-Filter" \
#--output-path "XGBClassifier-text-classifier.json" \
#--huggingface-token None \
#--repo-id "userName/modelName" \
#--plot-confusion-matrix True
#
# How to use the trained model:
#
#from transformers import AutoTokenizer, AutoModel
#from xgboost import XGBClassifier
#import torch.nn.functional as F
#import torch
#
#Mean Pooling - Take attention mask into account for correct averaging
#def mean_pooling(model_output, attention_mask):
#    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
# Load model from HuggingFace Hub
#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
#embedding_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
#device = ("cuda" if torch.cuda.is_available() else "cpu")
#embedding_model.to(device)
#
#bst = XGBClassifier({'device': device})
#bst.load_model('/path/to/XGBClassifier-text-classifier.json')
#
#def score_text(text, model):
#    # Get the encoded input
#    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#
#    # Forward pass
#    with torch.no_grad():
#        model_output = embedding_model(**encoded_input)
#
#    # Perform pooling and normalization
#    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
#
#    embedding = F.normalize(sentence_embedding, p=2, dim=1).numpy()
#    score = model.predict(embedding)[0]
#
#    return score
#
#score_text("Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente.", bst)
