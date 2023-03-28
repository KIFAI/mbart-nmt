import os, glob
import datasets
import pandas as pd
from time import perf_counter as pc
from datasets import Dataset
from tqdm import tqdm

from .utils import *
import math


class NmtDataLoader:
    def __init__(self, mbart_tokenizer, preprocessor, corpus_path, packing, packing_size=256, hybrid=True, group_key="domain"):
        self.tokenizer = mbart_tokenizer
        self.preprocessor = preprocessor
        self.src_lang, self.tgt_lang = self.preprocessor.properties["src_lang"], self.preprocessor.properties["tgt_lang"]
        self.max_token_length = self.preprocessor.properties["max_token_length"]

        train_corpus_path = glob.glob(os.path.join(corpus_path, "train_*.tsv"))[0]
        valid_corpus_path = glob.glob(os.path.join(corpus_path, "valid_*.tsv"))[0]
        self.train_dataset = self.get_parallel_dataset(train_corpus_path, packing, packing_size, hybrid, group_key)
        self.eval_dataset = self.get_parallel_dataset(valid_corpus_path, packing, packing_size, hybrid, group_key)
        self.raw_datasets = datasets.DatasetDict({"train": self.train_dataset, "valid": self.eval_dataset})

    def get_parallel_dataset(self, corpus_path, packing, packing_size, hybrid, group_key, header=["domain", "subdomain", "ko_KR", "en_XX"]):
        """
        Load splited src&tgt lang's corpus into huggingface dataset format
        """
        category_data = []
     
        corpus = pd.read_csv(corpus_path, sep="\t")
        print(corpus.head())

        src_data, tgt_data = corpus[self.src_lang], corpus[self.tgt_lang]

        if packing and (packing_size is not None):
            print("Merge sentences into Segments...")
            packed_src, packed_tgt, packed_len, time_num = packing_data(
                self.tokenizer, corpus, group_key, self.src_lang, self.tgt_lang, packing_size, self.max_token_length, merge_direction="bidirection"
            )
            
            if hybrid:
                print("Prepare train data using sents & segments unit")
                time_num_ceil = math.ceil(time_num)
                packed_src_data, packed_tgt_data = [], []
                for i in range(time_num_ceil):
                    packed_src_data.extend(shuffle_packed_data(i, packed_src))
                    packed_tgt_data.extend(shuffle_packed_data(i, packed_tgt))
                # src, tgt 맞는지 확인 필
                # 데이터를 증강 후 전체 src 길이로 자름
                print("len(packed_src_data) :", len(packed_src_data))
                print("len(packed_tgt_data) :", len(packed_tgt_data))
                packed_src_data = packed_src_data[: len(src_data)]
                packed_tgt_data = packed_tgt_data[: len(tgt_data)]
                print("src_data len :", len(src_data))

                src_data = [s.strip() for s in src_data]
                tgt_data = [s.strip() for s in tgt_data]
                src_data.extend([" ".join(sents) for sents in packed_src_data])
                tgt_data.extend([" ".join(sents) for sents in packed_tgt_data])
                
                random.seed(10)
                ixs = list(range(len(src_data)))
                random.shuffle(ixs)
                src_data = [src_data[i] for i in ixs]
                tgt_data = [tgt_data[i] for i in ixs]

                print("total len : ", len(src_data))

                # 테스트 코드
                sample = random.sample(range(len(src_data)), 15)
                print(sample)
                for i in sample:
                    print("*" * 100)
                    print(i)
                    print(src_data[i])
                    print(tgt_data[i])

            else:
                print("Prepare train data using only segments unit")
                src_data, tgt_data = [" ".join(sents) for sents in packed_src], [" ".join(sents) for sents in packed_tgt]
        else:
            print("No packing..")
            src_data, tgt_data = corpus[self.src_lang], corpus[self.tgt_lang]

        category_data = {
                f"{self.src_lang}" : [src_line.rstrip("\n") for src_line in src_data],
                f"{self.tgt_lang}" : [tgt_line.rstrip("\n") for tgt_line in tgt_data]
            }

        return Dataset.from_dict(category_data)

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

        inputs, targets = examples[src_lang], examples[tgt_lang]
        
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
