from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import argparse
import torch
import os

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir if args.cache_dir else "/tmp",
        token=args.token if args.token else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir if args.cache_dir else "/tmp",
        token=args.token if args.token else None,
        **{"attn_implementation":"sdpa"},
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the dataset (expected to be a dataset of json files)
    dataset_files = sorted([os.path.join(args.directory_path, f) for f in os.listdir(args.directory_path) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    dataset = load_dataset(
        "json",
        data_files=dataset_files,
        split='train',
        num_proc=len(dataset_files),
        cache_dir=args.cache_dir,
    )

    def foo(batch):

        encoded_input = tokenizer(
            batch[args.text_column],
            **{
            "padding" : True,
            "truncation" : True,
            "max_length" : 512, # The maximum length the BERTimbau model can handle
            "return_tensors" : "pt",
        }).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        logits = model_output.logits.cpu().numpy()
        labels = [int(np.argmax(logit)) for logit in logits]
        scores = [float(np.max(torch.softmax(torch.tensor(logit), dim=-1).numpy())) for logit in logits]

        return {"label": labels, "probs": scores}
    
    dataset = dataset.map(
        foo,
        batched=True,
        batch_size=args.batch_size if args.batch_size else 1,
        num_proc=1,
        desc="Classifying dataset",
    )

    os.makedirs(os.path.dirname(args.output_folder), exist_ok=True)

    indices = np.array_split(np.arange(len(d)), args.n_chunks)
    chunks = [d.select(idx) for idx in indices]

    n = 0
    for chunk in chunks:
        n += len(chunk)
    
    assert n == len(d), "The chunks are not of the expected length"

    # Save the chunks in disk
    for i, chunk in enumerate(chunks):
        chunk.to_json(
            os.path.join(output_dir, f"chunk_{i}.jsonl"), 
            num_proc=args.num_proc if args.num_proc else 1,
        )

    print(f"Saved dataset in '{args.output_folder}'.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be used.")
    parser.add_argument("--directory_path", type=str, required=True, help="The path to the directory containing the dataset.")
    parser.add_argument("--token", type=str, default=None, help="The token to access the dataset.")
    parser.add_argument("--cache_dir", type=str, default="/tmp", help="The directory to store the dataset.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the text column in the dataset.")
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument("--output_folder", type=str, required=True, help="The file to store the dataset.")
    parser.add_argument("--n_chunks", type=int, default=1, help="The number of chunks to split the dataset into.")

    args = parser.parse_args()

    main(args)

# python3 run-text-filter.py \
#--model_name "TucanoBR/BERTimbau-base-text-filter" \
#--directory_path "path/to/dataset" \
#--cache_dir "path/to/cache" \
#--token None \
#--batch_size 64 \
#--output_folder "path/to/output/folder" \
#--num_proc 16 \
#--n_chunks 20