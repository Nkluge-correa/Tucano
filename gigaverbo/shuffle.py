import os
import datasets
import argparse
import numpy as np

def main(args):

    test_size = args.test_size // 3

    train_dataset_files = sorted([os.path.join(args.directory_path_train, f) for f in os.listdir(args.directory_path_train) if f.endswith(".jsonl")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    dataset1 = datasets.load_dataset(
        "json",
        data_files=train_dataset_files[0:2],
        split='train',
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )
    dataset1 = dataset1.train_test_split(test_size=test_size, shuffle=True, seed=args.seed)

    dataset2 = datasets.load_dataset(
        "json",
        data_files=train_dataset_files[2:10],
        split='train',
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )
    dataset2 = dataset2.train_test_split(test_size=test_size, shuffle=True, seed=args.seed)

    dataset3 = datasets.load_dataset(
        "json",
        data_files=train_dataset_files[10:30],
        split='train',
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )
    dataset3 = dataset3.train_test_split(test_size=test_size, shuffle=True, seed=args.seed)

    train_dataset = datasets.concatenate_datasets([dataset1['train'], dataset2['train'], dataset3['train']])
    val_dataset = datasets.concatenate_datasets([dataset1['test'], dataset2['test'], dataset3['test']])

    print(train_dataset)
    print(val_dataset)

    indices = np.array_split(np.arange(len(train_dataset)), args.n_chunks)

    chunks = [train_dataset.select(idx) for idx in indices]

    n = 0
    for chunk in chunks:
        n += len(chunk)

    assert n == len(train_dataset), "The chunks are not of the expected length"

    for i, chunk in enumerate(chunks):
        chunk.to_json(
            os.path.join(args.directory_path_train, f"temp_chunk_{i}.jsonl"), 
            batch_size=args.batch_size,
            num_proc=args.num_proc,
        )

    for i, f in enumerate(train_dataset_files):
        os.remove(f)
        os.rename(os.path.join(args.directory_path_train, f"temp_chunk_{i}.jsonl"), f)

    os.makedirs(args.directory_path_val, exist_ok=True)

    val_dataset.to_json(
            os.path.join(args.directory_path_val, "0.jsonl"), 
            batch_size=args.batch_size,
            num_proc=args.num_proc,
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle the dataset")
    parser.add_argument('--directory_path_train', type=str, required=True, help="Path to the training data directory.")
    parser.add_argument('--test_size', type=int, default=60000, help="Size of the validation set.")
    parser.add_argument('--seed', type=int, default=1337, help="Seed for the random number generator.")
    parser.add_argument('--directory_path_val', type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument('--num_proc', type=int, default=64, help="Number of processes to use.")
    parser.add_argument('--batch_size', type=int, default=10000, help="Batch size for saving the dataset.")
    parser.add_argument('--cache_dir', type=str, default="/lustre/mlnvme/data/the_tucanos/cache", help="Path to the cache directory.")
    parser.add_argument('--n_chunks', type=int, default=30, help="Number of chunks to split the dataset into.")

    args = parser.parse_args()

    main(args)

# python shuffle.py \
# --directory_path_train "path/to/tokenized/gigaverbo/train" \
# --test_size 60000 \
# --seed 1337 \
# --directory_path_val "path/to/tokenized/gigaverbo/val" \
# --num_proc 64 \
# --batch_size 10000 \
# --cache_dir "path/to/cache" \
# --n_chunks 30