import argparse
import os
import torch
import tqdm
import json
from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main(args):
  
    # Set up precision and CUDA backend options
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, token=args.token, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16 if args.precision == "bfloat16" else torch.float32,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
    )

    # Set up generation configuration
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TextGenerationPipeline(model=model, task="text-generation", 
                                       tokenizer=tokenizer, device=device)

    # Load evaluation dataset
    eval_set = load_dataset("TucanoBR/alpaca-eval-pt", split="eval", cache_dir=args.cache_dir)
    eval_set = eval_set.remove_columns(["output", "generator"]) # These columns are not required for evaluation
    samples = eval_set['instruction'] # Get instruction samples

    # Generate outputs
    outputs = []
    
    for i in tqdm.tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[i:i + args.batch_size]
        batch = [f"{args.systems}{args.prompt_1}{sample}{args.prompt_2}" for sample in batch]

        completions = generator(batch, generation_config=generation_config)

        for sample, completion in zip(batch, completions):
            outputs.append(completion[0]['generated_text'][len(sample):])

    # Add output and generator columns to dataset and save
    generator_names = [args.model_id.split('/')[-1]] * len(outputs)
    eval_set = eval_set.add_column("output", outputs)
    eval_set = eval_set.add_column("generator", generator_names)
    output_file = os.path.join(args.output_folder, f"alpaca-eval-pt-{args.model_id.split('/')[-1]}.json")
    data = [sample for sample in eval_set]

    with open(output_file, 'w') as f:

        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error: {e}")
            json.dump(data, f, ensure_ascii=True, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a text generation model evaluation.")
    parser.add_argument("--model_id", type=str, default="TucanoBR/Tucano-2b4-Instruct", help="Model identifier")
    parser.add_argument("--revision", type=str, default="main", help="Model revision")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation")
    parser.add_argument("--precision", type=str, choices=["bfloat16", "float16"], default="bfloat16", help="Precision type")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for token repetition")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling for generation")
    parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save generated outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--systems", type=str, default="", help="Systems prompt")
    parser.add_argument("--prompt_1", type=str, default="<instruction>", help="Prompt 1")
    parser.add_argument("--prompt_2", type=str, default="</instruction>", help="Prompt 2")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    main(args)
