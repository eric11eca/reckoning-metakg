import re
import os
import torch
import random
import argparse

from pprint import pprint
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data.distributed import DistributedSampler

from .utils.py_io import read_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def kg_span_reconstruction(text):
    masked = text
    for i, token in enumerate(text.split(" ")):
        masked.replace(token, f"<extra_id_{i}>")
    return masked

class DataReader:
    """Custom dataset loader for prompt-response pairs."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        NotImplemented

    @classmethod
    def jsonl_file_reader(cls, path, config):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)
        if "context" in total_data[0]:
            all_facts = [fact for data in total_data for fact in data["context"]]
        elif "facts" in total_data[0]:
            all_facts = [fact for data in total_data for fact in data["facts"]]
        else:
            raise ValueError("No context or facts found in the data")

        dataset = []
        for instance in total_data:
            data = cls._read(instance, config)
            if config.random_facts:
                num_facts = len(data[0]["context"])
                data["context"] = random.choices(all_facts, k=num_facts)
                num_facts = len(data[0]["context"])
                data["context"] = random.choices(all_facts, k=num_facts)
            dataset.append(data)
        # pprint(dataset[0])
        # raise ValueError("Stop here")
        return dataset

class StudentRecordDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        context = instance["facts"]
        recall = instance["recall"]
        relation = instance["relation"]
        aggregation = instance["aggregation"]
        # problems = recall + relation + aggregation
        problems = [aggregation] * 2
        random.shuffle(context)

        qa_pairs = []
        if args.baseline:
            qa_pairs = [[p['question'], p['answer'], p.get('support', "")] for p in problems]
        if not args.baseline:
            prefix = f"record_0 to record_{len(context) - 1} are student records you memorized."
            for p in problems:
                q, a = p['question'], p['answer']
                qa_pairs.append([f"{prefix}\n{q}", a, p.get("support", "")])
            context = [f"record_{i}: {fact}" for i, fact in enumerate(context)]

        # if args.multi_task:
        #     criticals = "**\n".join(instance["facts"])
        #     for item in qa_pairs:
        #         item[1] = f"Based on\n{criticals}\nThe answer is: {item[1]}"

        return {"guid": guid, "qa_pairs": qa_pairs, "context": context}

class BabiLongDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        context = instance["facts"]
        problems = [{
            "question": instance["question"],
            "answer": instance["answer"],
            "support": instance.get("support", "")
        }]
        random.shuffle(context)

        qa_pairs = []
        if args.baseline:
            qa_pairs = [[p['question'], p['answer'], p['support']] for p in problems]
        if not args.baseline:
            prefix = f"record_0 to record_{len(context) - 1} are student records you memorized."
            for p in problems:
                q, a = p['question'], p['answer']
                qa_pairs.append([f"{prefix}\n{q}", a, p.get("support", "")])
            context = [f"record_{i}: {fact}" for i, fact in enumerate(context)]

        return {"guid": guid, "qa_pairs": qa_pairs, "context": context}

class LossInMidDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        context = instance["facts"]
        problems = [{
            'question': instance["question"],
            'answer': instance["answers"],
            'support': " ".join(instance.get("support", []))
        }]
        random.shuffle(context)

        qa_pairs = []
        if args.baseline:
            qa_pairs = [[p['question'], p['answer'], p['support']] for p in problems]
        if not args.baseline:
            prefix = f"record_0 to record_{len(context) - 1} are documents you memorized."
            for p in problems:
                q, a = p['question'], p['answer']
                qa_pairs.append([f"{prefix}\n{q}", a, p.get("support", "")])
            context = [f"record_{i}: {fact}" for i, fact in enumerate(context)]

        return {"guid": guid, "qa_pairs": qa_pairs, "context": context}

class ProofWriterDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        answer_map = {"true": "yes", "false": "no", "unknown": "none"}

        guid = instance["guid"]
        questions = instance["questions"]
        context = instance["facts"]
        answers = [answer_map[a] for a in instance["answers"]]
        qa_pairs = []

        if "distractors" in instance:
            criticals = instance["facts"]
            distractors = instance["distractors"]
            if args.load_order == "pre":
                context = criticals + distractors
            elif args.load_order == "post":
                context = distractors + criticals
            elif args.load_order == "in":
                context = distractors + criticals
                random.shuffle(context)

        if args.baseline:
            qa_pairs = [[q, a] for q, a in zip(questions, answers)]
        if not args.baseline:
            prefix = f"Given the the set of facts: from fact_0 to fact_{len(context) - 1}"
            for q, a in zip(questions, answers):
                qa_pairs.append([f"{prefix}, can we conclude {q}?", a])
            context = [f"fact_{i}: {fact}" for i, fact in enumerate(context)]

        if args.multi_task:
            criticals = "**\n".join(instance["facts"])
            for item in qa_pairs:
                item[1] = f"Based on\n{criticals}\nThe answer is: {item[1]}"

        return {"guid": guid, "qa_pairs": [item], "context": context}

class MetaKnowledgeDataset(Dataset):
    def __init__(self, args, tokenizer, data_path, data_type, is_training):
        self.args = args
        self.task = args.dataset

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.data_path = data_path
        self.data_type = data_type

        reader_classes = {
            "proofwriter": ProofWriterDataReader,
            "student_records": StudentRecordDataReader,
            "loss_in_mid": LossInMidDataReader,
            "babilong": BabiLongDataReader,
        }

        self.reader = reader_classes[args.dataset_type]
        self.is_training = is_training
        self.tokenizer = tokenizer

        self.data = self.read_data_from_file()
        if not self.is_training and args.max_eval_data > 0:
            self.data = random.choices(self.data, k=args.max_eval_data)

        if self.is_training and args.do_eval:
            self.data = self.data[:1]

        # if self.data_type == "test":
        #     self.data = self.data[:20]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        qa_data = self.data[index]
        if self.args.baseline:
            return self.causal_lm_base_collator(qa_data)
        else:
            return self.causal_lm_collator(qa_data)

    def read_data_from_file(self):
        file_path = f"{self.data_path}/{self.task}/{self.data_type}.jsonl"
        file_data = self.reader.jsonl_file_reader(file_path, self.args)
        return file_data

    def compute_seq_length(self, prompt, response):
        input_text = f"{prompt} {response}{self.tokenizer.eos_token}"
        tokenized_input = self.tokenizer.encode(input_text, add_special_tokens=True)
        return len(tokenized_input)

    def encode_with_prompt_completion_format(self, prompt, response, max_seq_length):
        '''
        Here we assume each example has 'prompt' and 'completion' fields.
        We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
        and it doesn't make sense to follow directly with the completion.
        '''
        example_text = f"{prompt} {response}{self.tokenizer.eos_token}"
        tokenized_example = self.tokenizer(
            example_text,
            return_tensors='pt',
            max_length=max_seq_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt = self.tokenizer(
            prompt, return_tensors='pt',
            max_length=max_seq_length,
            truncation=True
        )
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }

    def causal_lm_base_collator(self, qa_data):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        inputs, outputs = [], []
        for qa_pair in qa_data["qa_pairs"]:
            prompt = f"{qa_pair[0]}\nAnswer:"
            response = f"{qa_pair[1]}" # + f"\nSupport: {qa_pair[2]}"
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=True))
            response_len = len(self.tokenizer.encode(response, add_special_tokens=True))

            truncated_context = []
            total_context_len = 0
            for fact in qa_data["context"]:
                fact_len = len(self.tokenizer.encode(fact, add_special_tokens=False))
                if total_context_len + fact_len + prompt_len + response_len > 1000:
                    break
                truncated_context.append(fact)
                total_context_len += fact_len

            # print("total_context_len: ", total_context_len + prompt_len + response_len)

            context = "\n".join(truncated_context)
            prompt = f"{context}\n{prompt}"
            encoded = self.encode_with_prompt_completion_format(
                prompt, response, max_seq_length=1024)
            input_ids_batch.append(encoded['input_ids'])
            attention_mask_batch.append(encoded['attention_mask'])
            labels_batch.append(encoded['labels'])
            inputs.append(prompt)
            outputs.append(response)

        print_out = {
            "guid": qa_data["guid"],
            "prompt": inputs,
            "response": outputs,
        }

        return {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "labels": torch.stack(labels_batch, dim=0),
            "print_out": print_out,
            "task": self.task,
        }

    def causal_lm_collator(self, qa_data):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        train_input_ids_batch = []
        train_attention_mask_batch = []
        train_labels_batch = []
        train_inputs, train_outputs = [], []
        max_seq_length = 0
        sequences = []
        for fact in qa_data["context"]:
            if "fact_" in fact:
                pattern = r'fact_\d+:'
            else:
                pattern = r'record_\d+:'
            try:
                prompt = re.findall(pattern, fact)[0]
            except:
                prompt = fact.split(":")[0] + ":"
            splited = re.split(pattern, fact)
            response = splited[1]
            sequences.append((prompt, response))
            max_seq_length = max(max_seq_length, self.compute_seq_length(prompt, response))

        max_seq_length = min(max_seq_length, 128)

        # print("inner loop max_seq_length: ", max_seq_length)

        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt, response,
                max_seq_length=max_seq_length)
            train_input_ids_batch.append(encoded['input_ids'])
            train_attention_mask_batch.append(encoded['attention_mask'])
            train_labels_batch.append(encoded['labels'])
            train_inputs.append(prompt)
            train_outputs.append(response)

        dev_input_ids_batch = []
        dev_attention_mask_batch = []
        dev_labels_batch = []
        dev_inputs, dev_outputs = [], []
        sequences = []
        max_seq_length = 0
        for qa_pair in qa_data["qa_pairs"]:
            prompt = f"{qa_pair[0]}\nAnswer:"
            response = f"{qa_pair[1]}\nSupport: {qa_pair[2]}"
            sequences.append((prompt, response))
            max_seq_length = max(max_seq_length, self.compute_seq_length(prompt, response))

        # print("outer loop max_seq_length: ", max_seq_length)

        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt, response,
                max_seq_length=max_seq_length)
            dev_input_ids_batch.append(encoded['input_ids'])
            dev_attention_mask_batch.append(encoded['attention_mask'])
            dev_labels_batch.append(encoded['labels'])
            dev_inputs.append(prompt)
            dev_outputs.append(str(response))

        feature = {
            "input_ids": torch.stack(dev_input_ids_batch, dim=0),
            "attention_mask": torch.stack(dev_attention_mask_batch, dim=0),
            "labels": torch.stack(dev_labels_batch, dim=0),
            "train_input_ids": torch.stack(train_input_ids_batch, dim=0),
            "train_attention_mask": torch.stack(train_attention_mask_batch, dim=0),
            "train_labels": torch.stack(train_labels_batch, dim=0),
            "print_out": {"guid": qa_data["guid"]},
        }

        if not self.is_training:
            feature["print_out"].update({
                    "prompt": dev_inputs,
                    "response": dev_outputs
                }
            )
            feature["inner_print_out"] = {
                "guid": qa_data["guid"],
                "prompt": train_inputs,
                "response": train_outputs
            }
        return feature

def create_dataloader(args, dataset, is_training):
    if is_training:
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=unroll
    )

def unroll(batch):
   return batch[0]

# from transformers import AutoTokenizer

# def prepare(dataset, rank, world_size, batch_size=8, pin_memory=False, num_workers=0):
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

#     dataloader = DataLoader(
#         dataset, batch_size=batch_size,
#         pin_memory=pin_memory,
#         num_workers=num_workers,
#         drop_last=False, shuffle=False,
#         sampler=sampler)

#     return dataloader

# def setup(rank, world_size):
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

# def build_dataset(args):
#     tokenizer = AutoTokenizer.from_pretrained("susnato/phi-2")
#     tokenizer.pad_token = tokenizer.eos_token

#     train_dataset = MetaKnowledgeDataset(
#         args=args,
#         tokenizer=tokenizer,
#         data_path="./data",
#         data_type="train",
#         is_training=True,
#     )
#     sample = train_dataset[0]

#     train_input_ids = sample["train_input_ids"]
#     train_attention_mask = sample["train_attention_mask"]
#     train_labels = sample["train_labels"]

#     rebatch = [{
#         "input_ids": train_input_ids[i],
#         "attention_mask": train_attention_mask[i],
#         "labels": train_labels[i],
#     } for i in range(len(train_input_ids))]
#     return rebatch

# import time

# def main(rank, world_size, args):
#     print("rank", rank, "startup")
#     setup(rank, world_size)
#     dataset = build_dataset(args)
#     dataloader = prepare(dataset, rank, world_size)

#     time.sleep(5)

#     for epoch in range(1):
#         if rank == 0:
#             print("the current epoch is: ", epoch)
#         time.sleep(2)
#         dataloader.sampler.set_epoch(epoch)
#         for step, batch in enumerate(dataloader):
#             time.sleep(1)
#             print(f"rank: {rank}, step: {step}, batch: {batch['input_ids'][0][:10]}")
#             time.sleep(1)
#         if rank == 0:
#             print("the current epoch completes")
#         time.sleep(2)
#     dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="student_records")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_eval_data", type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--multi_task", action="store_true")
    parser.add_argument("--random_facts", action="store_true")
    args = parser.parse_args()

    # world_size = 8
    # mp.spawn(
    #     main,
    #     args=(world_size, args),
    #     nprocs=world_size
    # )

    # labels = sample["labels"][0]
    # labels[labels < 0] = 0
    # input_decoded = tokenizer.decode(sample["input_ids"][0], skip_special_tokens=True)
    # label_decoded = tokenizer.decode(labels, skip_special_tokens=True)
    # print(f"Input:\n{input_decoded}")
    # print(f"Label:\n{label_decoded}")

    # dataLoader = create_dataloader(args, train_dataset, is_training=True)
    # for batch in dataLoader:
    #     print(batch["input_ids"].shape)
    #     print(batch["labels"].shape)
    #     print(batch["attention_mask"].shape)
    #     print(batch["train_input_ids"].shape)
    #     print(batch["train_labels"].shape)
    #     print(batch["train_attention_mask"].shape)
    #     print(batch["print_out"])
    #     break

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    eval_dataset = MetaKnowledgeDataset(
        args=args,
        tokenizer=tokenizer,
        data_path="/main/data",
        data_type="val",
        is_training=False,
    )
    sample = eval_dataset[0]
    labels = sample["labels"][0]
    labels[labels < 0] = 0
    input_decoded = tokenizer.decode(sample["input_ids"][0], skip_special_tokens=True)
    label_decoded = tokenizer.decode(labels, skip_special_tokens=True)
    print(f"Input:\n{input_decoded}")
    print(f"Label:\n{label_decoded}")

    dataLoader = create_dataloader(args, eval_dataset, is_training=False)
    for batch in dataLoader:
        print(batch["input_ids"].shape)
        print(batch["labels"].shape)
        print(batch["attention_mask"].shape)
        print(batch["train_input_ids"].shape)
        print(batch["train_labels"].shape)
        print(batch["train_attention_mask"].shape)
        print(batch["print_out"])
        break