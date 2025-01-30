import argparse
from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import torch
import tqdm
import re
import os

def clean_next_word(next_word):
    return re.sub(r'[.,?!]', '', next_word).strip()

def main(args):

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    dataset = load_dataset(
        "TucanoBR/lambada-pt",
        split="train",
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )
    text_column = "sentence"
    target_column = "last_word"

    if args.n_shots > 0:
        dataset = dataset.train_test_split(test_size=args.n_shots, shuffle=False)
        examples = dataset["test"]
        dataset = dataset["train"]

        print(f"Using {args.n_shots} examples for evaluation")
        
        example_prompt = f"{args.system_prompt}"
        for sample in examples:
            example_prompt += f"{args.prompt_1}{sample[text_column]}{args.prompt_2}{sample[args.target_column]}{args.eos_token}\n\n"

        print("### Example prompt ###")
        print(f"""'''{example_prompt}'''""")
    
    else:
        print("### Example prompt ###")
        sample = next(iter(dataset))[text_column]
        example_prompt = f"{args.system_prompt}{args.prompt_1}{sample}{args.prompt_2}"
        print(f"""'''{example_prompt}'''""")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, use_auth_token=args.token, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.revision, use_auth_token=args.token, attn_implementation=args.attn_implementation, torch_dtype=precision)
    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TextGenerationPipeline(model=model, task="text-generation",
                                       tokenizer=tokenizer, device=device)

    samples = dataset[text_column]
    targets = dataset[target_column]

    predictions = []

    for i in tqdm.tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[i:i + args.batch_size]
        batch = [f"{example_prompt}{args.prompt_1}{sample}{args.prompt_2}" for sample in batch]

        completions = generator(batch, generation_config=generation_config) 

        for sample, completion in zip(batch, completions):
            try:
                next_word = completion[0]['generated_text'][len(sample):].split()[0]
            except IndexError:
                next_word = ""
            predictions.append(clean_next_word(next_word))

    acc = sum([1 for pred, target in zip(predictions, targets) if pred == target]) / len(targets)

    results ={
        "model": args.model_name,
        "revision" : args.revision,
        "metric": "accuracy",
        "calame_pt": acc
    }

    output_file = os.path.join(args.output_path, f"{args.model_name.split('/')[-1]}-revision-{args.revision}-lambada-pt.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text generation model on LAMBADA-PT dataset")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--revision', type=str, required=True, help='Model revision')
    parser.add_argument('--attn_implementation', type=str, default="sdpa", help='Attention implementation')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Hugging Face API token')
    parser.add_argument('--num_proc', type=int, default=8, help='Number of processes to use for loading dataset')
    parser.add_argument('--system_prompt', type=str, default="", help='System prompt')
    parser.add_argument('--prompt_1', type=str, default="", help='Prompt 1')
    parser.add_argument('--prompt_2', type=str, default="", help='Prompt 2')
    parser.add_argument('--eos_token', type=str, default="", help='End of sentence token')
    parser.add_argument('--n_shots', type=int, default=0, help='Number of examples to use for evaluation')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of new tokens to generate')
    parser.add_argument('--tf32', type=bool, default=False, help='Allow TF32')
    parser.add_argument('--bf16', type=bool, default=False, help='Allow BF16')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output_path', type=str, default=".", help='Output path')

    args = parser.parse_args()
    main(args)

