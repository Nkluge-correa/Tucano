import os
import json
import argparse
from transformers import AutoTokenizer
from huggingface_hub import create_repo, HfApi
from datasets import load_dataset, load_from_disk, concatenate_datasets

ADDITIONAL_SPECIAL_TOKENS = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|filename|>",
    "<|gh_stars|>",
    "<|issue_start|>",
    "<|issue_comment|>",
    "<|issue_closed|>",
    "<|jupyter_start|>",
    "<|jupyter_text|>",
    "<|jupyter_code|>",
    "<|jupyter_output|>",
    "<|empty_output|>",
    "<|commit_before|>",
    "<|commit_msg|>",
    "<|commit_after|>",
    "<|reponame|>",
    "<|user_input|>",
    "<|assistant_output|>",
    ]

def main(args):

    train_dataset_files = sorted([os.path.join(args.train_dataset_dir, f) for f in os.listdir(args.train_dataset_dir) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    dataset = load_dataset(
        "json",
        data_files=train_dataset_files,
        split='train',
        num_proc=len(train_dataset_files),
        cache_dir=args.cache_dir,
    )

    print("Dataset loaded.")

    dataset = dataset.remove_columns([col for col in dataset.column_names if col != args.text_column])
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    
    def batch_iterator():
        for i in range(0, len(dataset), args.batch_size):
            yield dataset[i : i + args.batch_size][args.text_column]

    if args.train_new_from_old:

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print("Training the tokenizer...")

        new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=args.vocab_size, new_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
        new_tokenizer.name_or_path = args.hub_tokenizer_name
        new_tokenizer.model_max_length = args.max_length

        print(f"Tokenizer trained. Saving it to disk in {args.output_dir}...")
        new_tokenizer.save_pretrained(args.output_dir)
        print(len(new_tokenizer))

        assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=False)
        assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        assert len(AutoTokenizer.from_pretrained(args.output_dir)) == args.vocab_size
        assert AutoTokenizer.from_pretrained(args.output_dir).model_max_length == args.max_length

        print("Pushing the tokenizer to the hub...")
        # Push the folder to the hub.
        create_repo(
            repo_id=args.hub_tokenizer_name, 
            token=args.token,
            repo_type="model",
            exist_ok=True,
            private=True
        )

        api = HfApi(token=args.token)

        api.upload_folder(
            repo_id=args.hub_tokenizer_name,
            folder_path=args.output_dir,
        )

        print("Tokenizer uploaded to the hub.")
    
    else:

        from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
        from transformers import GPT2TokenizerFast

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        print("Training the tokenizer...")
        trainer = trainers.BpeTrainer(vocab_size=args.vocab_size -1, special_tokens=ADDITIONAL_SPECIAL_TOKENS) # -1 because the tokenizer adds the <|endoftext|> token by default.
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()

        print(f"Tokenizer trained. Saving it to disk in {args.output_dir}...")
        new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
        new_tokenizer.save_pretrained(args.output_dir)
        new_tokenizer.name_or_path = args.hub_tokenizer_name
        new_tokenizer.model_max_length = args.max_length

        assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=False)
        assert AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        assert len(AutoTokenizer.from_pretrained(args.output_dir)) == args.vocab_size
        assert AutoTokenizer.from_pretrained(args.output_dir).model_max_length == args.max_length

        print("Pushing the tokenizer to the hub...")

        create_repo(
            repo_id=args.hub_tokenizer_name,
            token=args.token,
            repo_type="model",
            exist_ok=True,
            private=True
        )

        api = HfApi(token=args.token)

        api.upload_folder(
            repo_id=args.hub_tokenizer_name,
            folder_path=args.output_dir,
        )

        print("Tokenizer uploaded to the hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new BPE tokenizer")
    parser.add_argument("--train_dataset_dir", type=str, help="The path of the dataset (stored locally as jsonl files) to use for training the tokenizer.")
    parser.add_argument("--text_column", type=str, help="The name of the text column in the dataset.")
    parser.add_argument("--seed", type=int, help="The random seed used when shuffling the dataset.")
    parser.add_argument("--num_samples", type=int, help="The number of samples to use for training the tokenizer.")
    parser.add_argument("--batch_size", type=int, help="The batch size to use for training the tokenizer.")
    parser.add_argument("--vocab_size", type=int, help="The vocabulary size of the tokenizer.")
    parser.add_argument("--output_dir", type=str, help="The directory to save the trained tokenizer.")
    parser.add_argument("--hub_tokenizer_name", type=str, help="The name of the tokenizer in the hub.")
    parser.add_argument("--token", type=str, help="The hub token to use to upload the tokenizer.")
    parser.add_argument("--train_new_from_old", type=bool, help="Whether to train a new tokenizer from an already trained one.")
    parser.add_argument("--cache_dir", type=str, help="The directory to cache the dataset.")
    parser.add_argument("--max_length", type=int, help="The maximum length of the tokenizer.")
    args = parser.parse_args()

    main(args)

# python3 train-bpe-tokenizer.py \
#--train_dataset_dir "path/to/dataset" \
#--text_column "text" \
#--seed 1337 \
#--num_samples 2000000 \
#--batch_size 10000 \
#--vocab_size 32000 \
#--output_dir "./new-tokenizer" \
#--hub_tokenizer_name "username/new-BPE-tokenizer" \
#--token None \
#--train_new_from_old True \
#--cache_dir "path/to/cache" \
#--max_length 2048