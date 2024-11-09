import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
from xgboost import XGBRegressor
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
    """Preprocess the data to create a regression dataset"""
    embeddings_array = np.array(dataset['embedding'].tolist())
    embedding_df = pd.DataFrame(embeddings_array)
    embedding_df['label'] = pd.to_numeric(dataset['score'])
    return embedding_df

def split_data(embedding_df):
    """Split the data into train and test sets"""
    train, test = train_test_split(embedding_df, test_size=0.1)
    regression_x_train = train.iloc[:, :-1]
    regression_y_train = train['label']
    regression_x_test = test.iloc[:, :-1]
    regression_y_test = test['label']
    return regression_x_train, regression_y_train, regression_x_test, regression_y_test

def train_model(regression_x_train, regression_y_train, regression_x_test, regression_y_test, learning_rate, max_depth, n_estimators, early_stopping_rounds):
    """Train the XGBoost Regressor"""
    bst_r = XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        booster='gbtree',
        early_stopping_rounds=early_stopping_rounds,
    )
    bst_r.fit(
    regression_x_train, regression_y_train,
    eval_set=[(regression_x_test, regression_y_test)],
    verbose=100
    )
    return bst_r

def save_and_upload_model(bst_r, output_path, huggingface_token, repo_id):
    """Save and upload the trained model to the Hugging Face Hub"""
    bst_r.save_model(output_path)
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

    regression_x_train, regression_y_train, regression_x_test, regression_y_test = split_data(embedding_df)

    print("Train dataset: ", regression_x_train.shape)
    print("Train labels: ", regression_y_train.shape)
    print("Test dataset: ", regression_x_test.shape)
    print("Test labels: ", regression_y_test.shape)
    
    bst_r = train_model(regression_x_train, regression_y_train, regression_x_test, regression_y_test,
                        args.learning_rate, args.max_depth, args.n_estimators, args.early_stopping_rounds)
    preds = bst_r.predict(regression_x_test)
    
    predictions = bst_r.predict(regression_x_test)
    mse = mean_squared_error(regression_y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'MSE:{mse}')
    print(f'RMSE: {rmse}')
    
    save_and_upload_model(bst_r, args.output_path, args.huggingface_token, args.repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload a text regression model to the Hugging Face Hub")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset in the Hugging Face Hub")
    parser.add_argument('--learning-rate', type=float, default=0.1, help="Learning rate for training")
    parser.add_argument('--max-depth', type=int, default=2, help="Maximum depth of the trees")
    parser.add_argument('--n-estimators', type=int, default=1000, help="Number of boosting rounds")
    parser.add_argument('--early-stopping-rounds', type=int, default=300, help="Early stopping rounds")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--huggingface-token', type=str, required=True, help="Hugging Face token for authentication")
    parser.add_argument('--repo-id', type=str, required=True, help="Hugging Face repo id to upload the model")
    args = parser.parse_args()
    main(args)

# How to run this script:
#
# python train-xgboost-regressor.py --dataset-path "MulaBR/GigaVerbo-text-classifier" \
#--output-path "XGBRegressor-text-classifier.json" \
#--huggingface-token None \
#--repo-id "userName/modelName"
#
#
# How to use the trained model:
#
#from transformers import AutoTokenizer, AutoModel
#from xgboost import XGBRegressor
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
#bst_r = XGBRegressor({'device': device})
#bst_r.load_model('/path/to/XGBRegressor-text-classifier.json')
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
#score_text("Os tucanos são aves que correspondem à família Ramphastidae, vivem nas florestas tropicais da América Central e América do Sul. A família inclui cinco gêneros e mais de quarenta espécies diferentes. Possuem bicos notavelmente grandes e coloridos, que possuem a função de termorregulação para as muitas espécies que passam muito tempo na copa da floresta exposta ao sol tropical quente.", bst_r)