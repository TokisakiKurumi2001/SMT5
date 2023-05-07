from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from transformers import T5Tokenizer
import random

class SMT5QADataLoader:
    def __init__(self, ckpt: str, query_max_length: int, answer_max_length: int):
        data_dict = {'train': 'data/mkqa.train.csv'}
        dataset = load_dataset('csv', data_files=data_dict)
        self.tokenizer = T5Tokenizer.from_pretrained(ckpt)
        self.query_max_length = query_max_length
        self.answer_max_length = answer_max_length
        random.seed(42)

        # self.dataset = dataset

        self.dataset = dataset.shuffle(seed=42).map(
            self.__tokenize,
            remove_columns=dataset["train"].column_names
        )

    def __tokenize(self, examples):
        rt_dict = {}
        query_toks = self.tokenizer(examples['query'], max_length=self.query_max_length, truncation=True, padding="max_length")
        answer_toks = self.tokenizer(examples['answer'], max_length=self.answer_max_length, truncation=True, padding="max_length")
        for k, v in query_toks.items():
            rt_dict[f'query_{k}'] = v
        for k, v in answer_toks.items():
            rt_dict[f'answer_{k}'] = v
        rt_dict['idx'] = examples['idx']
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=1)
            )
        return res

class SMT5TSNLIDataLoader:
    def __init__(self, ckpt: str, max_length: int):
        data_dict = {'train': 'data/tsnli.train.csv'}
        dataset = load_dataset('csv', data_files=data_dict)
        self.tokenizer = T5Tokenizer.from_pretrained(ckpt)
        self.max_length = max_length
        random.seed(42)

        # self.dataset = dataset

        self.dataset = dataset.shuffle(seed=42).map(
            self.__tokenize,
            remove_columns=dataset["train"].column_names
        )

    def __tokenize(self, examples):
        rt_dict = {}
        anchor_toks = self.tokenizer(examples['anchor'], max_length=self.max_length, truncation=True, padding="max_length")
        ent_toks = self.tokenizer(examples['entail'], max_length=self.max_length, truncation=True, padding="max_length")
        con_toks = self.tokenizer(examples['contract'], max_length=self.max_length, truncation=True, padding="max_length")
        for k, v in anchor_toks.items():
            rt_dict[f'anchor_{k}'] = v
        for k, v in ent_toks.items():
            rt_dict[f'pos_{k}'] = v
        for k, v in con_toks.items():
            rt_dict[f'neg_{k}'] = v
        rt_dict['idx'] = examples['idx']
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=1)
            )
        return res
