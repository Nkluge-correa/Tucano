import os
import json
import argparse
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import create_repo, HfApi
from transformers import LlamaTokenizerFast, LlamaTokenizer, AutoTokenizer


def main(args):

    dataset_file = args.dataset_file # Path to the dataset file. The SentencePieceTrainer requires a .txt file as input.

    # Check if the dataset file exists.
    if not os.path.exists(args.dataset_file):

        # Load the dataset from the huggingface Hub and prepare it for training.
        if args.train_dataset_dir is not None:

            train_dataset_files = sorted([os.path.join(args.train_dataset_dir, f) for f in os.listdir(args.train_dataset_dir) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

            dataset = load_dataset(
                "json",
                data_files=train_dataset_files,
                split='train',
                num_proc=len(train_dataset_files),
                cache_dir=args.cache_dir,
            )
        else:
            raise ValueError("No dataset directory provided. Please provide a dataset directory to train the tokenizer.")

        dataset = dataset.remove_columns([col for col in dataset.column_names if col != args.text_column])

        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
        print("Number of samples selected from the dataset:", len(dataset))

        with open(args.dataset_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset):
                f.write(example["text"])
    
    else:
        print("Dataset file already exists. Skipping the dataset preparation step.")
    
    print("Training the tokenizer...")

    ## Learn more about the arguments of `SentencePieceTrainer` in [here](https://github.com/google/sentencepiece/blob/master/doc/options.md).
    spm.SentencePieceTrainer.Train(
        input=args.dataset_file,
        model_prefix='tokenizer',
        vocab_size=args.vocab_size,
        unk_id=0,
        unk_piece="<unk>",
        bos_id=1,
        bos_piece="<s>",
        eos_id=2,
        eos_piece="</s>",
        model_type='bpe',
        byte_fallback=True,
    )

    tokenizer = LlamaTokenizer("./tokenizer.model")

    # Add the "<pad>" token. Why? Read [this](https://huggingface.co/docs/transformers/main/model_doc/llama2#usage-tips).
    tokenizer.add_special_tokens(
        {
            "pad_token":"<pad>"
        }
    )

    # Add other special tokens if needed.
    ADDITIONAL_SPECIAL_TOKENS = [
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
    "<user_input>",
    "<assistant_output>",
    ]

    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

    # Ideally, your tokenizer should have a vocab size of `vocab_size + len(ADDITIONAL_SPECIAL_TOKENS) + 1` (for the "<pad>" token).
    # This in the end, for optimal performance, should be a power of 2, or at least a multiple of 2.
    print("LlamaTokenizer vocab size:", len(tokenizer))
    tokenizer.save_pretrained("./new-llama-tokenizer")

    tokenizer = LlamaTokenizerFast("./new-llama-tokenizer/tokenizer.model")
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
    print("LlamaTokenizerFast vocab size:", len(tokenizer))
    tokenizer.save_pretrained("./new-llama-tokenizer")

    with open("./new-llama-tokenizer/tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)
    
    tokenizer_config['legacy'] = False
    tokenizer_config['bos_token_id'] = tokenizer.bos_token_id
    tokenizer_config['eos_token_id'] = tokenizer.eos_token_id
    tokenizer_config['pad_token_id'] = tokenizer.pad_token_id
    tokenizer_config['unk_token_id'] = tokenizer.unk_token_id
    tokenizer_config['padding_side'] = "right"

    with open("./new-llama-tokenizer/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=4)
    
    assert AutoTokenizer.from_pretrained("./new-llama-tokenizer/", use_fast=False)
    assert AutoTokenizer.from_pretrained("./new-llama-tokenizer/", use_fast=True)

    # Push the folder to the hub.
    create_repo(
        repo_id=args.tokenizer_name, 
        token=args.token,
        repo_type="model",
        exist_ok=True,
        private=True
    )

    api = HfApi(token=args.token)

    api.upload_folder(
        repo_id=args.tokenizer_name,
        folder_path="./new-llama-tokenizer/",
    )

    print("Tokenizer uploaded to the hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new Llama tokenizer")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="The path to the dataset file",
    )
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        default=None,
        help="The path to the dataset directory",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the text column in the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed used when shuffling the dataset",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The token to access the dataset on the hub",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory to cache the dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2_000_000,
        help="Number of samples to use from the dataset. You might run into memory issues if you use too many samples.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size to use for the tokenizer",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Name of the tokenizer to be uploaded to the hub",
    )
    args = parser.parse_args()
    main(args)

# python3 train-sentencepiece.py \
#--dataset_file "path/to/tokenizer-dataset.txt" \
#--train_dataset_dir "path/to/GigaVerbo" \
#--text_column "text" \
#--seed 1337 \
#--token None \
#--cache_dir "path/to/cache" \
#--num_samples 20000000 \
#--vocab_size 32747 \
#--tokenizer_name "your_user/your_tokenizer_name"