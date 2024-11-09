import os
import mteb
import json
import torch
import argparse
import numpy as np
from typing import Any
from llm2vec import LLM2Vec
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with custom or base models.")
    parser.add_argument("--base_model_name_or_path", type=str, default="TucanoBr/Tucano-1b1-mntp",
                        help="Path to the base model.")
    parser.add_argument("--peft_model_name_or_path", type=str, default=None,
                        help="Path to the PEFT model.")
    parser.add_argument("--is_custom", action="store_true", 
                        help="Flag to indicate if using a custom model.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save output results.")
    return parser.parse_args()

def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction

tasks_and_instructions = {
    "MassiveIntentClassification": "Dada a descrição da tarefa para qual um produto será utilizado, classifique o produto:",
    # "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages",
    # "mteb/amazon_massive_intent" "https://arxiv.org/abs/2204.08582"
    
    "MassiveScenarioClassification": "Dada a descrição da tarefa para qual um produto será utilizado, classifique o produto:",
    # "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages"
    # "mteb/amazon_massive_scenario" "https://arxiv.org/abs/2204.08582"
    
    "MultiHateClassification": "Classifique essa sentença como contendo ou não discurso de ódio:",
    # Hate speech detection dataset with binary (hateful vs non-hateful) labels. Includes 25+ distinct types of hate and challenging non-hate, and 11 languages.
    # "mteb/multi-hatecheck" "https://aclanthology.org/2022.woah-1.15/"
    
    "SIB200Classification": "Dado um trecho de texto, classifique ele de acordo com o tópico:",
    # SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects annotated. The dataset is annotated in English for the topics,  science/technology, travel, politics, sports, health, entertainment, and geography. The labels are then transferred to the other languages in Flores-200 which are machine-translated.
    # "mteb/sib200" "https://arxiv.org/abs/2309.07445"

    "TweetSentimentClassification": "Dado um tweet em português, classifique-o de acordo com sua emoção (positivo, negativo ou neutro):",
    # "A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages."
    # "mteb/tweet_sentiment_multilingual" "https://aclanthology.org/2022.lrec-1.27"

    "HateSpeechPortugueseClassification": "Classifique essa sentença como contendo ou não discurso de ódio:",
    # HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).
    # "hate-speech-portuguese/hate_speech_portuguese" "https://aclanthology.org/W19-3510"

    "MintakaRetrieval": "Dada uma pergunta, avalie se uma dada resposta é relevante para a pergunta:",
    # We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers.
    # "jinaai/mintakaqa" "https://aclanthology.org/2022.coling-1.138"
    
    "WikipediaRetrievalMultilingual": "Dada uma pergunta, avalie se uma dada resposta é relevante para a pergunta:",
    # The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.
    # "ellamind/wikipedia-2023-11-retrieval-multilingual-queries" "https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries"
    
    "Assin2RTE": "Dada duas sentenças, avalie o grau de similaridade entre elas:",
    # Recognizing Textual Entailment part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.
    # "nilc-nlp/assin2" "https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39"

    "STSBenchmarkMultilingualSTS": "Dada duas sentenças, avalie o grau de similaridade entre elas:",
    # Semantic Textual Similarity Benchmark (STSbenchmark) dataset but translated using DeepL API.
    # "mteb/stsb_multi_mt" "https://github.com/PhilipMay/stsb-multi-mt/"

    "Assin2STS": "Dada duas sentenças, avalie o grau de similaridade entre elas:"
    # Semantic Textual Similarity part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.
    # "nilc-nlp/assin2" "https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39"
}


class LLM2VecWrapper:
    def __init__(self, model=None, tasks_and_instructions=None):
        self.model = model
        self.tasks_and_instructions = tasks_and_instructions

    def encode(self, sentences: list[str], *, prompt_name: str = None, **kwargs: Any) -> np.ndarray:
        if prompt_name is not None:
            instruction = llm2vec_instruction(self.tasks_and_instructions.get(prompt_name, ""))
        else:
            instruction = ""
        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

def main():
    args = parse_args()
    
    if args.is_custom:

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
        config = AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Loading MNTP (Masked Next Token Prediction) model.
        model = PeftModel.from_pretrained(
            model,
            args.base_model_name_or_path,
        )

        if args.peft_model_name_or_path is not None:

            # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
            model = model.merge_and_unload()
            model = PeftModel.from_pretrained(
                model,
                args.peft_model_name_or_path,
            )

        # Wrapper for encoding and pooling operations
        model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        model = LLM2VecWrapper(model=model, tasks_and_instructions=tasks_and_instructions)
    else:
        model = SentenceTransformer(args.base_model_name_or_path)
    
    tasks = [
        mteb.get_task(task, languages = ["por"]) for task in tasks_and_instructions.keys()

    ]

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)
    print("Evaluation complete. Results saved to:", args.output_dir)

    task_subsets = {
        "Assin2RTE": "test",
        "Assin2STS": "test",
        "HateSpeechPortugueseClassification": "train",
        "MassiveIntentClassification": "test",
        "MassiveScenarioClassification": "test",
        "MintakaRetrieval": "test",
        "MultiHateClassification": "test",
        "SIB200Classification": "test",
        "STSBenchmarkMultilingualSTS": "dev",
        "TweetSentimentClassification": "test",
        "WikipediaRetrievalMultilingual": "test"
    }

    first_subdir = os.listdir(args.output_dir)[0]
    second_subdir = os.listdir(os.path.join(args.output_dir, first_subdir))[0]
    evals_path = os.path.join(args.output_dir, first_subdir, second_subdir)

    results = {}

    for task_name, subset in task_subsets.items():
        task_eval_path = os.path.join(evals_path, task_name + ".json")
        with open(task_eval_path, "r") as f:
            task_eval = json.load(f)
        results[task_name] = task_eval['scores'][subset][0]["main_score"]
                
    with open(evals_path + "/results.json", "w") as f:
        json.dump(results, f)

    average_score = sum(score for score in results.values() if score is not None) / len(results)

    markdown_table = f"| Model | Average | {' | '.join(results.keys())} |\n"
    markdown_table += f"|-------|{'------|' * len(results)}------|\n"
    model_row = f"| {args.base_model_name_or_path.split('/')[-1]} " + f" | {average_score:.4f} |"
    model_row += " | ".join(f"{results[task]:.4f}" if results[task] is not None else "N/A" for task in results.keys()) + " |\n"
    markdown_table += model_row
    print(markdown_table)

    dataset_types = {
        "MassiveIntentClassification": "Classification",
        "MassiveScenarioClassification": "Classification",
        "MultiHateClassification": "Classification",
        "SIB200Classification": "Classification",
        "TweetSentimentClassification": "Classification",
        "HateSpeechPortugueseClassification": "Classification",
        "Assin2RTE": "PairClassification",
        "MintakaRetrieval": "Retrieval",
        "WikipediaRetrievalMultilingual": "Retrieval",
        "STSBenchmarkMultilingualSTS": "STS",
        "Assin2STS": "STS"
    }

    # Create a table with the results averaged by dataset type
    dataset_results = {}
    for task_name, score in results.items():
        dataset_type = dataset_types[task_name]
        if dataset_type not in dataset_results:
            dataset_results[dataset_type] = []
        dataset_results[dataset_type].append(score)

    dataset_averages = {dataset_type: sum(scores) / len(scores) for dataset_type, scores in dataset_results.items()}

    markdown_table = f"| Model | Average | {' | '.join(dataset_averages.keys())} |\n"
    markdown_table += f"|-------|{'------|' * len(dataset_averages)}------|\n"
    model_row = f"| {args.base_model_name_or_path.split('/')[-1]} " + f" | {average_score:.4f} |"
    model_row += " | ".join(f"{dataset_averages[dataset]:.4f}" for dataset in dataset_averages.keys()) + " |\n"
    markdown_table += model_row
    print(markdown_table)

if __name__ == "__main__":
    main()

#pip install llm2vec -q
#pip install llm2vec[evaluation] -q

#python mteb-custom-eval.py \
#--base_model_name_or_path "TucanoBr/Tucano-1b1-mntp" \
#--peft_model_name_or_path "TucanoBR/Tucano-1b1-mntp-simcse" \
#--is_custom \
#--output_dir "results"