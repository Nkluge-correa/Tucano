from transformers import AutoTokenizer
import numpy as np
import datasets
import argparse
import os

DATASET_REPEAT = {
    "corpus_carolina" : 1,
    "blogset" : 1,
    "hplt_pt_0" : 1,
    "crawlPT_dedup_0" : 1,
    "hplt_pt_1" : 1,
    "crawlPT_dedup_1" : 1,
    "hplt_pt_2" : 1,
    "crawlPT_dedup_2" : 1,
    "hplt_pt_3" : 1,
    "crawlPT_dedup_3" : 1,
    "cc_2023" : 1,
    "mc4_pt" : 1,
    "culturax" : 1,
    "wikipedia" : 1,
    "gpt4all" : 1,
    "bactrianx" : 1,
    "dolly_15k" : 1,
    "instruct_ptbr" : 1, 
    "roots_ted" : 1,
    "roots_wiki" : 1,
    "xlsum" : 1,
    "ultrachat" : 1,
    "legal_pt" : 1,
    "cosmos_qa" : 1,
}

def main(args):

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the tokenizer we will use to tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else "TucanoBR/Tucano-1b1",
        cache_dir=args.cache_dir if args.cache_dir else "/cache",
        token=args.token if args.token else None,
        use_fast=True,
    )

    print(f"Loaded tokenizer {args.tokenizer_name}")

    # Define a function to tokenize the text
    def tokenize(examples):

        # Tokenize a sample and return them as a list of token ids, we don't need the attention mask or token type ids for our model
        input_ids = tokenizer(examples['text'], return_attention_mask=False, return_token_type_ids=False)
        
        # Add to the beginning of each example the end-of-sequence token
        input_ids['input_ids'] = [[tokenizer.eos_token_id] + sublist for sublist in input_ids['input_ids']]
        
        return input_ids

    block_size = args.block_size

    # Define a function to group the texts together so that we have chunks of `block_size`
    def group_texts(examples):
        """ Group texts together so that we have chunks of `block_size` """

        # Concatenate tokens from examples for each key in the examples dictionary
        concatenated_examples = {
            k: [t for example in examples[k] for t in example ] for k in examples.keys()
        }

        # Calculate the total length of the concatenated tokens for the first key in examples.
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # If the total length is greater than or equal to the block size, adjust it to be a multiple of the block size.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split the concatenated tokens into blocks of size `block_size`.
        result = {
            # For each key and token list in concatenated examples
            k: [
                # Create a sublist for each block of size `block_size`
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }

        # Return the processed blocks of tokens.
        return result
    
    # Create an enpty list to store all portions of GigaVerbo
    text_datasets = []

    # Iterate over the dataset names and the number of times to repeat them
    for dataset_name, n_repeat in DATASET_REPEAT.items():

        # Load the dataset subset from disk 
        path = os.path.join(args.datasets_dir, dataset_name)

        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

        dataset = datasets.load_dataset(
            "json",
            data_files=files,
            split='train',
            num_proc=args.num_proc if args.num_proc else 1,
            cache_dir=args.cache_dir if args.cache_dir else "/cache",
        )

        # Repeat the dataset n_repeat times
        dataset = datasets.concatenate_datasets([dataset for _ in range(n_repeat)])

        # Add the dataset to the list
        text_datasets.append(dataset)
        print(f"Loaded dataset '{dataset_name}' with {len(dataset):,} examples") 
        print(f"Dataset was repeated {n_repeat} times (original size: {len(dataset) // n_repeat:,} examples)")
    
    # Concatenate all the text datasets
    full_dataset = datasets.concatenate_datasets(text_datasets)

    # Delete the text datasets and last dataset to free up memory
    del text_datasets
    del dataset

    print(f"Concatenated all datasets | {len(full_dataset):,} examples")

    full_dataset = full_dataset.map(
        tokenize,
        batched=True,
        remove_columns=full_dataset.column_names,
        desc=f"Running tokenizer on every text in dataset",
        num_proc=args.num_proc if args.num_proc else 1,
        load_from_cache_file=True,
    )
    
    full_dataset = full_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {args.block_size}",
        num_proc=args.num_proc if args.num_proc else 1,
        load_from_cache_file=True,
    )

    print(f"Train dataset has {len(full_dataset):,} examples ({len(full_dataset) *  args.block_size:,} tokens)")
    print(full_dataset)

    # Create the output directory for the tokenized dataset
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    # Given that the train dataset can be quite large, we save it in chunks os size `n_chunks`
    indices = np.array_split(np.arange(len(full_dataset)), args.n_chunks)

    # Split the dataset into `num_chunks` chunks using the indices
    chunks = [full_dataset.select(idx) for idx in indices]

    # Assert that the chunks are of the expected length
    n = 0
    for chunk in chunks:
        n += len(chunk)
    
    assert n == len(full_dataset), "The chunks are not of the expected length"

    # Save the chunks in disk
    for i, chunk in enumerate(chunks):
        chunk.to_json(
            os.path.join(args.output_dir, "train", f"chunk_{i}.jsonl"), 
            num_proc=args.num_proc if args.num_proc else 1,
        )

    print(f"Tokenized dataset saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset for Causal Language Modeling")
    parser.add_argument("--output-dir", type=str, help="Output directory to save the tokenized dataset")
    parser.add_argument("--cache-dir", type=str, help="Cache directory to store the tokenizer")
    parser.add_argument("--datasets-dir", type=str, help="Directory containing the datasets to tokenize")
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use")
    parser.add_argument("--block-size", type=int, help="Block size to use")
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--num-proc", type=int, help="Number of processes to use")
    parser.add_argument("--n-chunks", type=int, default=20, help="Number of chunks to split the train dataset into")
    parser.add_argument("--seed", type=int, default=1337, help="Seed to use for shuffling the dataset")

    args = parser.parse_args()
    main(args)

# python3 tokenize-gigaverbo.py \
#--output-dir "/lustre/mlnvme/data/the_tucanos/gigaverbo-tokenized" \
#--cache-dir "/lustre/mlnvme/data/the_tucanos/cache" \
#--datasets-dir "/lustre/mlnvme/data/the_tucanos/gigaverbo" \
#--tokenizer-name "TucanoBR/Tucano-1b1" \
#--block-size 2048 \
#--token None \
#--num-proc 64 \
#--seed 1337 \
#--n-chunks 30
