import argparse
from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import torch
import tqdm
import re

def clean_next_word(next_word):
    return re.sub(r'[.,?!]', '', next_word).strip()

def main(args):

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16
    precision = torch.bfloat16 if args.bf16 else torch.float32

    dataset = load_dataset("TucanoBR/lambada-pt", split="train")
    dataset = dataset.to_pandas()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, use_auth_token=args.token)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.revision, use_auth_token=args.token, attn_implementation="sdpa", torch_dtype=precision)
    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TextGenerationPipeline(model=model, task="text-generation",
                                       tokenizer=tokenizer, device=device)

    predictions = []

    for sentence in tqdm.tqdm(dataset.sentence):
        completion = generator(sentence, generation_config=generation_config)
        
        try:
            next_word = completion[0]['generated_text'][len(sentence):].split()[0]
        except IndexError:
            next_word = ""

        predictions.append(clean_next_word(next_word))

    dataset['predictions'] = predictions
    dataset['acc'] = (dataset['predictions'] == dataset['last_word']).astype(int)
    acc = dataset['acc'].sum()/ len(dataset)

    results ={
        "model": f"{args.model_name}-{args.revision}",
        "lambada_pt": acc
    }
    output_file = f"{args.output_path}/lambada_pt.json"
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Accuracy on LAMBADA-PT: {acc * 100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text generation model on LAMBADA-PT dataset")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--revision', type=str, required=True, help='Model revision')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of new tokens to generate')
    parser.add_argument('--tf32', type=bool, default=False, help='Allow TF32')
    parser.add_argument('--bf16', type=bool, default=False, help='Allow BF16')
    parser.add_argument('--output_path', type=str, default=".", help='Output path')

    args = parser.parse_args()
    main(args)

# python lm-lambada-pt-eval.py \
#--model_name "TucanoBR/Tucano-160m" \
#--revision "main" \
#--token None \
#--max_new_tokens 10 \
#--tf32 True \
#--bf16 True \
#--output_path "path/to/output"
