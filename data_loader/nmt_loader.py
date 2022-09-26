import os
import datasets
import pandas as pd
from time import perf_counter as pc
from datasets import Dataset
from tqdm import tqdm

from .utils import *


class NmtDataLoader:
    def __init__(self, mbart_tokenizer, preprocessor, corpus_path, packing, packing_size=256, hybrid=True):
        self.tokenizer = mbart_tokenizer
        self.preprocessor = preprocessor
        self.src_lang, self.tgt_lang = self.preprocessor.properties["src_lang"], self.preprocessor.properties["tgt_lang"]
        self.max_token_length = self.preprocessor.properties["max_token_length"]

        self.train_dataset = self.get_parallel_dataset(corpus_path, "train", packing, packing_size, hybrid)
        self.eval_dataset = self.get_parallel_dataset(corpus_path, "valid", packing, packing_size, hybrid)
        self.raw_datasets = datasets.DatasetDict({"train": self.train_dataset, "validation": self.eval_dataset})

    def get_parallel_dataset(self, corpus_path, category, packing, packing_size, hybrid):
        """
        Load splited src&tgt lang's corpus into huggingface dataset format
        """
        category_data = []
        src_path = os.path.join(corpus_path, category)
        tgt_path = os.path.join(corpus_path, category)

        with open(f"{src_path}.{self.src_lang}", "r") as src, open(f"{tgt_path}.{self.tgt_lang}", "r") as tgt:
            src_data = src.readlines()
            tgt_data = tgt.readlines()

        if packing and (packing_size is not None):
            print('Merge sentences into Segments...')
            packed_src, packed_tgt, packed_len = packing_data(self.tokenizer, src_data, tgt_data, packing_size, self.max_token_length, merge_direction='bidirection')
            if hybrid:
                print('Prepare train data using sents & segments unit')
                src_data = [s.strip() for s in src_data] + [" ".join(sents) for sents in packed_src] * 6
                tgt_data = [s.strip() for s in tgt_data] + [" ".join(sents) for sents in packed_tgt] * 6
                '''
                seen, seen_src, seen_tgt = set(), {}, {}
                dupes_ix = []
                uniq_src, uniq_tgt = [], []

                for i, sent in tqdm(enumerate(zip(src_data, tgt_data)), total=len(src_data)):
                    if sent[1] in seen:
                        pass
                    else:
                        seen.add(sent[1])
                        seen_src[i], seen_tgt[i] = sent[0], sent[1]

                src_data, tgt_data = list(seen_src.values()), list(seen_tgt.values())
                assert len(src_data) == len(tgt_data)
                print(f"Duplicate cases # : {len(dupes_ix)}:")
                print(f"Uniq sent & segments unit # : {len(src_data)}")
                '''
            else:
                print('Prepare train data using only segments unit')
                src_data, tgt_data = [" ".join(sents) for sents in packed_src], [" ".join(sents) for sents in packed_tgt]
        else:
            print('No packing..')

        for i, lines in enumerate(tqdm(zip(src_data, tgt_data), total=len(src_data))):
            category_data.append(
                {
                    "translation": {
                        f"{self.src_lang}": lines[0].rstrip("\n"),
                        f"{self.tgt_lang}": lines[1].rstrip("\n"),
                    }
                }
            )
        return Dataset.from_pandas(pd.DataFrame(category_data))

    def get_tokenized_dataset(self, batch_size=20000, num_proc=8):
        self.tokenized_datasets = self.raw_datasets.map(
            self.preprocessor.preprocess,
            batched=True,
            batch_size=batch_size,
            remove_columns=self.raw_datasets["train"].column_names,
            fn_kwargs=self.preprocessor.properties,
            num_proc=num_proc,
        )

        return self.tokenized_datasets


class Processor:
    def __init__(self, mbart_tokenizer, src_lang="en_XX", tgt_lang="ko_KR", max_token_length=512, drop=True, bi_direction=True):
        """
        Argments for huggingface dataset's user defined map function
        Ref) During pre-training, mbart use instance format of up to 512 tokens
        """
        self.properties = {
            "mbart_tokenizer": mbart_tokenizer,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "max_token_length": max_token_length,
            "drop_case": drop,
            "bi_direction": bi_direction,
        }

    @staticmethod
    def preprocess(examples, mbart_tokenizer, src_lang, tgt_lang, max_token_length, drop_case, bi_direction):
        """
        User defined map function for huggingface dataset
        """
        mbart_tokenizer.src_lang = src_lang
        mbart_tokenizer.tgt_lang = tgt_lang

        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        if drop_case:
            model_inputs = mbart_tokenizer(inputs)

            # Setup the tokenizer for targets
            with mbart_tokenizer.as_target_tokenizer():
                labels = mbart_tokenizer(targets)

            srcs_ids, srcs_attn, tgts_ids, drop_case = [], [], [], []

            for src, src_attn, tgt in zip(model_inputs.input_ids, model_inputs.attention_mask, labels.input_ids):
                if len(src) < max_token_length and len(tgt) < max_token_length:
                    if bi_direction:
                        srcs_ids.extend([src, tgt])
                        srcs_attn.extend([src_attn, [1] * len(tgt)])
                        tgts_ids.extend([tgt, src])
                        drop_case.append(src)
                    else:
                        srcs_ids.append(src)
                        srcs_attn.append(src_attn)
                        tgts_ids.append(tgt)
                        drop_case.append(src)

            model_inputs["input_ids"] = srcs_ids
            model_inputs["labels"] = tgts_ids
            model_inputs["attention_mask"] = srcs_attn

        else:
            model_inputs = mbart_tokenizer(inputs, max_length=max_token_length, truncation=True)

            # Setup the tokenizer for targets
            with mbart_tokenizer.as_target_tokenizer():
                labels = mbart_tokenizer(targets, max_length=max_token_length, truncation=True)

            srcs_ids, srcs_attn, tgts_ids, drop_case = [], [], [], []

            for src, src_attn, tgt in zip(model_inputs.input_ids, model_inputs.attention_mask, labels.input_ids):
                if bi_direction:
                    srcs_ids.extend([src, tgt])
                    srcs_attn.extend([src_attn, [1] * len(tgt)])
                    tgts_ids.extend([tgt, src])
                    drop_case.append(src)
                else:
                    srcs_ids.append(src)
                    srcs_attn.append(src_attn)
                    tgts_ids.append(tgt)
                    drop_case.append(src)

            model_inputs["input_ids"] = srcs_ids
            model_inputs["labels"] = tgts_ids
            model_inputs["attention_mask"] = srcs_attn

        return model_inputs
