import os
import datasets
import itertools
import pandas as pd
import numpy as np
from time import perf_counter as pc
from datasets import Dataset
from tqdm import tqdm
from bashplotlib.histogram import plot_hist

class NmtDataLoader:
    def __init__(self, mbart_tokenizer, preprocessor, corpus_path):
        self.tokenizer = mbart_tokenizer
        self.preprocessor = preprocessor
        self.src_lang, self.tgt_lang = self.preprocessor.properties['src_lang'], self.preprocessor.properties['tgt_lang']
        self.max_token_length = self.preprocessor.properties['max_token_length']

        self.train_dataset = self.get_parallel_dataset(corpus_path, category='train')
        self.eval_dataset = self.get_parallel_dataset(corpus_path, category='valid')
        self.raw_datasets = datasets.DatasetDict({"train": self.train_dataset, "validation": self.eval_dataset})
    
    def _batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    def _packing_data(self, src_data, tgt_data, batch_size):
        start = pc()
        src_batched = [(s, [len(tokens) for tokens in self.tokenizer(s, add_special_tokens=True).input_ids]) for s in self._batch(src_data, batch_size)]
        src_data, src_lens = list(itertools.chain(*[item[0] for item in src_batched])), list(itertools.chain(*[item[1] for item in src_batched]))

        tgt_batched = [(t, [len(tokens) for tokens in self.tokenizer(t, add_special_tokens=True).input_ids]) for t in self._batch(tgt_data, batch_size)]
        tgt_data, tgt_lens = list(itertools.chain(*[item[0] for item in tgt_batched])), list(itertools.chain(*[item[1] for item in tgt_batched]))
        end = pc()
        print(f"Elapsed time for tokenizing batched src & tgt data : {end-start}")

        assert len(src_data) == len(tgt_data) == len(src_lens) == len(tgt_lens)
        print("Distribution of source sentense's len")
        plot_hist(src_lens, bincount=100)
        print("Distribution of target sentense's len")
        plot_hist(tgt_lens, bincount=100)

        parallel_data = sorted(zip(src_data, tgt_data, src_lens, tgt_lens),key = lambda item:item[2], reverse = False)
        print(f"Len of parallel data : {len(parallel_data)}")

        trigger, src_len, tgt_len = 0, 0, 0
        packed_src, packed_tgt, packed_len = [], [], []
        joined_src, joined_tgt = [], []

        for src, tgt, src_token_num, tgt_token_num in tqdm(parallel_data, total=len(parallel_data)):
            sent_len = src_token_num if src_token_num > tgt_token_num else tgt_token_num

            if trigger + sent_len > self.max_token_length:
                packed_src.append(joined_src)
                packed_tgt.append(joined_tgt)
                packed_len.append([src_len,tgt_len])

                joined_src, joined_tgt = [], []
                joined_src.append(src)
                joined_tgt.append(tgt)
                src_len, tgt_len = src_token_num, tgt_token_num

                trigger = sent_len
            else:
                joined_src.append(src)
                joined_tgt.append(tgt)
                src_len += src_token_num
                tgt_len += tgt_token_num
                trigger += sent_len

        print(f"Packed efficiency : {np.array(packed_len).mean(axis=0)} / {np.array(packed_len).mean(axis=0)/self.max_token_length}")
        print(f"Len of packed data : {len(packed_src)}, {len(packed_tgt)}")
        print(f"*****Quantile of packing sents : {np.quantile([item[0] for item in packed_len], [0.1, 0.25, 0.5, 0.75, 0.9, 1])}*****")
        plot_hist([item[0] for item in packed_len], bincount=100)
        return packed_src, packed_tgt, packed_len

    def get_parallel_dataset(self, corpus_path, category='train', packing=True, batch_size=256):
        '''
        Load splited src&tgt lang's corpus into huggingface dataset format
        '''
        category_data = []
        src_path = os.path.join(corpus_path, category)
        tgt_path = os.path.join(corpus_path, category)

        with open(f"{src_path}.{self.src_lang}", "r") as src, open(f"{tgt_path}.{self.tgt_lang}", "r") as tgt:
            src_data = src.readlines()
            tgt_data = tgt.readlines()

        if packing and (batch_size is not None):
            packed_src, packed_tgt, packed_len = self._packing_data(src_data, tgt_data, batch_size)
            src_data, tgt_data = [' '.join(sents) for sents in packed_src], [' '.join(sents) for sents in packed_tgt]
        else:
            pass

        for i, lines in enumerate(tqdm(zip(src_data, tgt_data), total=len(src_data))):
            category_data.append(
                {
                    "translation": {
                        f"{self.src_lang}": lines[0].rstrip('\n'),
                        f"{self.tgt_lang}": lines[1].rstrip('\n'),
                    }
                }
            )
        return Dataset.from_pandas(pd.DataFrame(category_data))

    def get_tokenized_dataset(self, batch_size=20000, num_proc=8):
        self.tokenized_datasets = self.raw_datasets.map(self.preprocessor.preprocess, batched=True, 
                batch_size=batch_size, remove_columns=self.raw_datasets["train"].column_names, 
                fn_kwargs=self.preprocessor.properties, num_proc=num_proc)

        return self.tokenized_datasets


class Processor:
    def __init__(self, mbart_tokenizer, src_lang='en_XX', tgt_lang='ko_KR', max_token_length=512,
            drop=True, bi_direction=True):
        '''
        Argments for huggingface dataset's user defined map function
        Ref) During pre-training, mbart use instance format of up to 512 tokens
        '''
        self.properties = {"mbart_tokenizer": mbart_tokenizer,
                           "src_lang": src_lang,
                           "tgt_lang": tgt_lang,
                           "max_token_length": max_token_length,
                           "drop_case":drop,
                           "bi_direction":bi_direction}

    @staticmethod
    def preprocess(examples, mbart_tokenizer, src_lang, tgt_lang, max_token_length, drop_case, bi_direction):
        '''
        User defined map function for huggingface dataset
        '''
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
                if len(src) < max_token_length and len(tgt) < max_token_length :
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

            model_inputs['input_ids'] = srcs_ids
            model_inputs['labels'] = tgts_ids
            model_inputs['attention_mask'] = srcs_attn

        else:
            model_inputs = mbart_tokenizer(
                inputs, max_length=max_token_length, truncation=True)

            # Setup the tokenizer for targets
            with mbart_tokenizer.as_target_tokenizer():
                labels = mbart_tokenizer(
                    targets, max_length=max_token_length, truncation=True)

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

            model_inputs['input_ids'] = srcs_ids
            model_inputs['labels'] = tgts_ids
            model_inputs['attention_mask'] = srcs_attn

        return model_inputs
