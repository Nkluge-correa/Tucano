import os
import json
import tqdm
import shutil
import argparse
import numpy as np
from datasets import load_dataset

# All portions of GigaVerbo
all_datasets = {
    "hplt_pt" : "source: https://huggingface.co/datasets/HPLT/hplt_monolingual_v1_2", # 58244012
    "wikipedia" : "source: https://huggingface.co/datasets/graelo/wikipedia", # 1101475
    "culturax" : "source: https://huggingface.co/datasets/uonlp/CulturaX", # 999994
    "gpt4all" : "source: https://huggingface.co/datasets/pablo-moreira/gpt4all-j-prompt-generations-pt", # 808803
    "bactrianx" : "source: https://huggingface.co/datasets/MBZUAI/Bactrian-X", # 66994
    "crawlPT_dedup" : "source: https://huggingface.co/datasets/eduagarcia/CrawlPT_dedup", #  cc100_pt (39236152) + oscar (1097388) + brwac (3513434)
    "dolly_15k" : "source: https://huggingface.co/datasets/Gustrd/dolly-15k-libretranslate-pt", # 28401 
    "instruct_ptbr" : "source: https://huggingface.co/datasets/cnmoro/Instruct-PTBR-ENUS-11M", # 2962856 
    "cosmos_qa" : "source: https://huggingface.co/datasets/heloisy/cosmos_qa_ptbr", # 25260
    "roots_ted" : "source: https://huggingface.co/datasets/bigscience-data/roots_pt_ted_talks_iwslt", # 3950
    "roots_wiki" : "source: https://huggingface.co/datasets/bigscience-data/roots_pt_wikiquote", # 6790
    "xlsum" : "source: https://huggingface.co/datasets/csebuetnlp/xlsum", # 64577
    "mc4_pt" : "source: https://huggingface.co/datasets/thegoodfellas/mc4-pt-cleaned", # 16092571
    "blogset" : "source: https://huggingface.co/datasets/thegoodfellas/blogset-br", # 4321181
    "cc_2023" : "source: https://huggingface.co/datasets/dominguesm/CC-MAIN-2023-23", # 12470998
    "ultrachat" : "source: https://huggingface.co/datasets/recogna-nlp/UltrachatBR", # 1255091
    "corpus_carolina" : "source: https://huggingface.co/datasets/carolina-c4ai/corpus-carolina", # 2075395
    "legal_pt" : "source: https://huggingface.co/datasets/eduagarcia/LegalPT_dedup", # 925522
}

def main(args):

    # Download GigaVerbo
    dataset = load_dataset(
        args.dataset_name, 
        split=args.dataset_split, 
        token=args.token if args.token else None,
        cache_dir=args.cache_dir if args.cache_dir else "/tmp",
        num_proc=args.num_proc if args.num_proc else 1,
    )

    # Loop over all dataset portions
    for dataset_name, dataset_meta_tag in all_datasets.items():

        # Create a folder to store the dataset using `args.output_dir` and `dataset_name`
        output_dir = os.path.join(args.output_dir, dataset_name)
        
        # Check if the folder already exists. If exists, remove it.
        if os.path.exists(output_dir):
            print(f"{dataset_name} already exists. Removing it...")
            shutil.rmtree(output_dir)

        print(f"Creating '{output_dir}' folder...")
        os.makedirs(output_dir)

        # Filter the dataset using the dataset meta tag
        d = dataset.filter(
            lambda example: example["metadata"] == dataset_meta_tag,
            num_proc=args.num_proc if args.num_proc else 1,
            desc=f"Parsing {dataset_name}"
        )

        print(f"Parsed '{dataset_name}' dataset.")
        print("Original dataset size:", len(d))
        o = len(d)

        # Filter the dataset using the label and probs columns
        d = d.filter(
                # Remove all rows that have a label other than 1 (1 is high quality, 0 is low quality)
                lambda example: example["label"] == 1 or (example["label"] == 0 and example["probs"] < args.confidence_threshold),
                num_proc=args.num_proc if args.num_proc else 1,
                desc=f"""Filtering {dataset_name}""",
            )
        
        print("Filtered dataset size:", len(d))
        print("Percentage of the original dataset: {:.2f}%".format((len(d) / o) * 100))

        # Remove the label, metadata, and probs columns
        d = d.remove_columns(["label", "probs", "metadata"])

        # Create indices for the chunks
        indices = np.array_split(np.arange(len(d)), args.n_chunks)

        # Split the dataset into `num_chunks` chunks using the indices
        chunks = [d.select(idx) for idx in indices]

        # Assert that the chunks are of the expected length
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

        print(f"Saved '{dataset_name}' dataset.")
    
    print("All datasets were saved.")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to be downloaded.")
    parser.add_argument("--dataset_split", type=str, required=True, help="The split of the dataset to be downloaded.")
    parser.add_argument("--token", type=str, default=None, help="The token to access the dataset.")
    parser.add_argument("--cache_dir", type=str, default="/tmp", help="The directory to store the dataset.")
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to store the dataset.")
    parser.add_argument("--confidence_threshold", type=float, default=0.98, help="The confidence threshold to filter the dataset.")
    parser.add_argument("--n_chunks", type=int, default=20, help="The number of chunks to split the dataset into.")

    args = parser.parse_args()

    main(args)

# python3 split-gigaverbo.py \
# --dataset_name "TucanoBR/GigaVerbo" \
# --dataset_split "train" \
# --token None \
# --cache_dir "path/to/cache" \
# --num_proc 64 \
# --output_dir "path/to/gigaverbo" \
# --confidence_threshold 0.98 \
# --n_chunks 20

            