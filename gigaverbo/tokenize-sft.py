from transformers import AutoTokenizer
from typing import  Dict, Union, List
from abc import ABC, abstractmethod
from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import argparse
import random
import torch
import copy
import os

def main(args):

    # Download the `Tucano-SFT` dataset from the Hugging Face Hub
    dataset = load_dataset(
        "TucanoBR/Tucano-SFT", 
        split='train', 
        token=args.token if args.token else None,
        cache_dir=args.cache_dir if args.cache_dir else "/tmp",
        num_proc=args.num_proc if args.num_proc else 1,
    )

    # Add the system prompt to the beginning of the conversation if the system prompt is not None
    if args.system_prompt is not None and args.system_prompt != "":

        dataset = dataset.map(
            lambda x: {"conversations": [{"role": "system", "content":args.system_prompt}] + x["conversations"]}, 
            num_proc=args.num_proc if args.num_proc else 1,
            desc="Adding system prompt to conversations",
        )

    # Check if the user part is longer than the assistant part:
    # We use this infomration to decide whether to mask the user part or the assistant part.
    # This approach is based on the following study [https://arxiv.org/abs/2401.13586]. In short,
    # in samples where the prompt is larger than the completion, masking the prompt is seems to
    # improve the performance of the model. In samples where the completion is larger than the prompt,
    # prompts can have some regularizing effect on the model. Therefore, latter we will only mask the
    # prompt a certain percentage of the times. 
    # 
    # FYI, in the Tucano-SFT dataset:
    #- 413451 samples have the assistant part longer.
    #- 266158 samples have the user part longer.

    def compare_lengths(example):
        conversation = example["conversations"]
        user_content = next(item["content"] for item in conversation if item["role"] == "user")
        assistant_content = next(item["content"] for item in conversation if item["role"] == "assistant")
        return {"user_longer_than_assistant": len(user_content) > len(assistant_content)}

    dataset = dataset.map(
        compare_lengths, 
        num_proc=args.num_proc if args.num_proc else 1,
        desc="Comparing lengths",
    )

    # Download the tokenizer and define the chat template
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir if args.cache_dir else "/tmp")
    
    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template
        tokenizer.push_to_hub(args.tokenizer_name + "-" + str(tokenizer.model_max_length) + "-chat-template", private=True, token=args.token if args.token else None)

    # Below we have the definition of the data structure that will be used to format the chat messages.
    # Here, we are using the `\n Usuário: ` and `\n Assistente: ` to separate the user and assistant messages.
    SLOT = Union[str, List[str], Dict[str, str]]

    @dataclass
    class Formatter(ABC):
        slot: SLOT = ""

        @abstractmethod
        def apply(self, **kwargs) -> SLOT: ...

    @dataclass
    class EmptyFormatter(Formatter):
        def apply(self, **kwargs) -> SLOT:
            return self.slot

    @dataclass
    class StringFormatter(Formatter):
        def apply(self, **kwargs) -> SLOT:
            msg = ""
            for name, value in kwargs.items():
                if value is None:
                    msg = self.slot.split(':')[0] + ":"
                    return msg
                if not isinstance(value, str):
                    raise RuntimeError("Expected a string, got {}".format(value))
                msg = self.slot.replace("{{" + name + "}}", value, 1)
            return msg

    format_user: "Formatter" = StringFormatter(slot="\n Usuário" + ": " + "{{content}}")
    format_assistant: "Formatter" = StringFormatter(slot="\n Assistente" + ": " + "{{content}}" + tokenizer.eos_token)
    system: "Formatter" = EmptyFormatter(slot=args.system_prompt if args.system_prompt is not None else "")
    separator: "Formatter" = EmptyFormatter(slot=['\n Assistente: ', tokenizer.eos_token])

    def get_list_from_message(messages):
        """
        This function receives a list of messages and returns two lists: one with the questions and another with the answers.

        messages  ====>  [{role:user, content:message}, {role:assistant, content:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['role'] != 'user':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['content'])
            else:
                answer_list.append(message['content'])

        assert len(question_list) == len(answer_list) , \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list

    def create_prompt(
            question_list, answer_list
        ):
        """
        This function receives a list of questions and a list of answers and returns a formatted string with the chat messages.

        question_list  ====>  ["question1", "question2", "question3"]
        answer_list  ====>  ["answer1", "answer2", "answer3"]
        """
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += system.apply()
            msg += format_user.apply(content=question)
            msg += format_assistant.apply(content=answer)
        return msg

    def make_masks(labels, tokenizer, sep, eos_token_length, rounds, mask):
        """
        This function receives the labels, the tokenizer, the separator, the eos_token_length, the rounds, 
        and a mask flag and returns the labels with the instruction part masked. Masking is done only 
        if the mask flag is True or if a random number is less than the THRESHOLD.
        """
        # Get a random number between 0.0 and 1.0
        rand = random.random()
        
        # It will mask the instruction part of the prompt if the mask flag is True or if the random number is less than the THRESHOLD.
        # For example, if the threshold is 0.8, the instruction part will be masked 80% of the time if the mask flag is False.
        if mask or rand < args.threshold:
            #### Maskes the input_ids to ignore the instruction part of the prompt ####
            cur_len = 0 # no bos_token
            eos_token_length = 1
            labels[:cur_len] = args.ignore_index
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(tokenizer.encode(rou)) + eos_token_length
                instruction_len = len(tokenizer.encode(parts[0])) - 1
                labels[0][cur_len:(cur_len + instruction_len)] = args.ignore_index     
                cur_len += round_len
        else:
            #### Labels are a copy of the input_ids ####
            cur_len = len(labels[0])

        return labels, cur_len

    def make_labels(input_ids, prompt, tokenizer, mask):
        """
        This function receives the input_ids, the prompt, the tokenizer, and a mask flag and returns the labels.
        """
        labels = copy.deepcopy(input_ids)
        sep, eos_token = separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = len(tokenizer.encode(eos_token))
        labels, cur_len = make_masks(labels, tokenizer, sep, eos_token_length, rounds, mask)
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)
                labels[:] = args.ignore_index
        return labels

    def encode(messages, tokenizer, mask):
        """
        This function receives a list of messages, a tokenizer, and a mask flag and returns the input_ids and the labels.
        It combines all previous functions to create the input_ids and the labels.
        """
        # Get the questions and answers from the messages
        question_list, answer_list = get_list_from_message(messages)

        # Create the prompt
        prompt = create_prompt(question_list, answer_list)

        # Tokenize the prompt and get the input_ids and the attention_mask
        # We pad the input_ids to the model_max_length and return the input_ids as a tensor
        encoding = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=tokenizer.model_max_length, 
            padding="max_length",
            truncation=True,
            )
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        if len(input_ids[0]) > tokenizer.model_max_length:
            input_ids = input_ids[:, :tokenizer.model_max_length]
            attention_mask = attention_mask[:, :tokenizer.model_max_length]

        # Create the labels
        labels = make_labels(input_ids, prompt, tokenizer, mask)

        # Change the labels with the value 3 [the pad_token_id] to 1 [the IGNORE_INDEX]
        # so we can ignore the pad_token_id when calculating the loss
        for j in range(len(labels[0])):
            if labels[0][j] == 3:
                labels[0][j] = args.ignore_index

        # Assert that the input_ids, attention_mask, and labels have the same length
        assert len(input_ids[0]) == tokenizer.model_max_length, f"input_ids have different lengths: {len(input_ids[0])} and {tokenizer.model_max_length}."
        assert len(labels[0]) == tokenizer.model_max_length, f"labels have different lengths: {len(labels[0])} and {tokenizer.model_max_length}."
        assert len(attention_mask[0]) == tokenizer.model_max_length, f"attention_mask have different lengths: {len(attention_mask[0])} and {tokenizer.model_max_length}."
        assert len(input_ids[0]) == len(labels[0]), f"input_ids and labels have different lengths: {len(input_ids)} and {len(labels)}."
        assert len(input_ids[0]) == len(attention_mask[0]), f"input_ids and attention_mask have different lengths: {len(input_ids)} and {len(attention_mask)}."

        return dict(
            input_ids=input_ids[0],
            attention_mask=attention_mask[0],
            labels=labels[0]
        )

    # Encode the dataset
    dataset = dataset.map(
        lambda x: encode(x["conversations"], tokenizer, mask=x["user_longer_than_assistant"]),
        num_proc=args.num_proc if args.num_proc else 1,
        desc="Encoding the dataset",
        remove_columns=dataset.column_names,
    )

    # Split the dataset into train and test. We are using only 1000 samples for testing.
    dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    print(dataset)

    # Create the output folder if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    # create internal "train" and "val" folders
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)

    train_dataset, val_dataset = dataset["train"], dataset["test"]

    for dataset, split in zip([train_dataset, val_dataset], ["train", "val"]):
        # Split the dataset into chunks
        indices = np.array_split(np.arange(len(dataset)), args.n_chunks if split == "train" else 1)
        chunks = [dataset.select(idx) for idx in indices]

        n = 0
        for chunk in chunks:
            n += len(chunk)

        assert n == len(dataset), "The chunks are not of the expected length"

        # Save the chunks in disk
        for i, chunk in enumerate(chunks):
            chunk.to_json(
                os.path.join(args.output_dir, split, f"chunk_{i}.jsonl"),
                num_proc=args.num_proc if args.num_proc else 1,
            )

        print(f"Saved the dataset in '{args.output_dir}/{split}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize the Tucano-SFT dataset.")
    parser.add_argument("--token", type=str, default=None, help="The Hugging Face Hub token.")
    parser.add_argument("--cache_dir", type=str, default="/cache", help="The cache directory.")
    parser.add_argument("--num_proc", type=int, default=64, help="The number of processors.")
    parser.add_argument("--system_prompt", type=str, default="", help="The system prompt.")
    parser.add_argument("--tokenizer_name", type=str, default="TucanoBR/Tucano-1b1", help="The tokenizer name.")
    parser.add_argument("--chat_template", type=str, default=None, help="The chat template.")
    parser.add_argument("--ignore_index", type=int, default=-100, help="The ignore index.")
    parser.add_argument("--threshold", type=float, default=0.8, help="The threshold. If threshold is 0.8, the instruction part will be masked 0.8 of the time (if the mask flag is False).")
    parser.add_argument("--test_size", type=int, default=1000, help="The test size.")
    parser.add_argument("--seed", type=int, default=1337, help="The seed.")
    parser.add_argument("--output_dir", type=str, default="tucano=stf", help="The output folder.")
    parser.add_argument("--n_chunks", type=int, default=5, help="The number of chunks.")
    args = parser.parse_args()
    main(args)

# python3 tokenize-sft.py \
#--token "your_token_here" \
#--cache_dir "/lustre/mlnvme/data/the_tucanos/cache" \
#--num_proc 64 \
#--system_prompt "Um bate-papo entre um usuário curioso e um assistente de inteligência artificial. O nome do assistente é Tucano, um grande modelo de linguagem desenvolvido para a geração de texto em português. O assistente dá respostas úteis, detalhadas e educadas em português às perguntas do usuário." \
#--tokenizer_name "TucanoBR/Tucano-1b1" \
#--chat_template "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n Usuário: ' + message['content'] }}{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n Assistente: '  + message['content'] + eos_token }}{% endif %}{% endfor %}" \
#--ignore_index -100 \
#--threshold 0.8 \
#--test_size 1000 \
#--seed 1337 \
#--output_dir "/lustre/mlnvme/data/the_tucanos/tucano-sft-2048" \
#--n_chunks 5
