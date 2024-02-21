import os, glob, re
import datasets
import pandas as pd
import logging
from time import perf_counter as pc
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from .utils import *
import math

class NmtDataLoader:
    def __init__(self, mbart_tokenizer, preprocessor, corpus_dir, packing, packing_size=256, hybrid=True, group_key="domain"):
        """
        Class to prepare parallel data for nmt task as huggingface dataset format
        Args :
            mbart_tokenizer : tokenizer class object
            preprocessor : preprocessor class object about huggingface dataset's user defined map function
            corpus_dir : director name of train or valid corpus to use in training
            packing : boolean value to choose which to use packing data
            packing_size : the accepted length to pack data
            hybird : the option to use sentence and segment(paragraph) together
            group_key : the key to use in merging(domain or subdomain)
        self.raw_datasets : example data is below.
            DatasetDict({
                ko_KR2en_XX: DatasetDict({
                    train: Dataset({
                        features: ['translation'],
                        num_rows: 169413
                    })
                    valid: Dataset({
                        features: ['translation'],
                        num_rows: 191
                    })
                })
                ko_KR2hi_IN: DatasetDict({
                    train: Dataset({
                        features: ['translation'],
                        num_rows: 313198
                    })
                    valid: Dataset({
                        features: ['translation'],
                        num_rows: 329
                    })
                })
                ko_KR2id_ID: DatasetDict({
                    train: Dataset({
                        features: ['translation'],
                        num_rows: 294891
                    })
                    valid: Dataset({
                        features: ['translation'],
                        num_rows: 319
                    })
                })
                ko_KR2km_KH: DatasetDict({
                    train: Dataset({
                        features: ['translation'],
                        num_rows: 291978
                    })
                    valid: Dataset({
                        features: ['translation'],
                        num_rows: 314
                    })
                })
                ko_KR2vi_VN: DatasetDict({
                    train: Dataset({
                        features: ['translation'],
                        num_rows: 304959
                    })
                    valid: Dataset({
                        features: ['translation'],
                        num_rows: 325
                    })
                })
            })
        """
        # set logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("INFO")
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s')
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        # Languages to support in this dataloader
        self.LANGS = ["en_XX", "ko_KR", "km_KH", "id_ID", "hi_IN", "vi_VN"]

        # The tokenizer and preprocessor object to use in dataloader
        self.tokenizer = mbart_tokenizer
        self.preprocessor = preprocessor

        # Validate rule of file name
        for path in glob.glob(f"{corpus_dir}/*"):
            fn = path.split("/")[-1]
            try:
                category, _, lang_pair, number_with_extension = fn.split("_")
            except:
                raise ValueError("File name should be ${train or valid}_corpus_${LANG}_${NUMBER}.tsv")
            
            if category not in ["train", "valid"]:
                raise ValueError("File name should be ${train or valid}_corpus_${LANG}_${NUMBER}.tsv")
            
            is_support, is_not_supported = [], []
            for support in self.LANGS :
                lang = support.split('_')[0]
                if lang in lang_pair:
                    is_support.append(lang)
                else:
                    is_not_supported.append(lang)

            if len(is_support) != 2:
                raise ValueError(f"The lang pair is {lang_pair}. {is_not_supported} don't be supported.")
            
            if re.sub(r'\d+', '', number_with_extension) != '.tsv':
                raise ValueError("File name should be ${train or valid}_corpus_${LANG}_${NUMBER}.tsv")

        # Load corpus paths of train and valid
        train_corpus_paths = sorted(glob.glob(os.path.join(corpus_dir, "train_corpus_*.tsv")))
        valid_corpus_paths = sorted(glob.glob(os.path.join(corpus_dir, "valid_corpus_*.tsv")))

        assert len(train_corpus_paths) == len(valid_corpus_paths)

        # get parallel huggingface dataset format 
        self.train_dataset_bundle = self.get_parallel_dataset(train_corpus_paths, packing, packing_size, hybrid, group_key)
        self.eval_dataset_bundle = self.get_parallel_dataset(valid_corpus_paths, packing, packing_size, hybrid, group_key)
        
        self.raw_datasets = datasets.DatasetDict()

        for train_corpus_path, valid_corpus_path in zip(train_corpus_paths, valid_corpus_paths):
            src2tgt = list(self.train_dataset_bundle[train_corpus_path].keys())[0]
            self.logger.info(f"\ntrain_corpus_path : {train_corpus_path}\nvalid_corpus_path : {valid_corpus_path}\nsource language : {src2tgt.split('2')[0]}\ntarget language : {src2tgt.split('2')[-1]}\n")
            
            train_dataset = self.train_dataset_bundle[train_corpus_path][src2tgt]
            valid_dataset = self.eval_dataset_bundle[valid_corpus_path][src2tgt]

            self.raw_datasets.update(
                                        {
                                            src2tgt : datasets.DatasetDict({"train": train_dataset, "valid": valid_dataset})
                                         }
                                    )

    def get_parallel_dataset(self, corpus_paths, packing, packing_size, hybrid, group_key):
        """
        Load splited src&tgt lang's corpus into huggingface dataset format
        Args :
            corpus_paths : absolute path of train or valid corpus to use in training
            packing : boolean value to choose which to use packing data
            packing_size : the accepted length to pack data
            hybird : the option to use sentence and segment(paragraph) together
            group_key : the key to use in merging(domain or subdomain)
        Returns:
            hf_dataset_bundle : example output is below.
                {'/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp/train_corpus_ko2en_149850.tsv': {'ko_KR2en_XX': Dataset({
                features: ['translation'],
                num_rows: 169413
            })}, '/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp/train_corpus_ko2hi_274479.tsv': {'ko_KR2hi_IN': Dataset({
                features: ['translation'],
                num_rows: 313198
            })}, '/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp/train_corpus_ko2id_274478.tsv': {'ko_KR2id_ID': Dataset({
                features: ['translation'],
                num_rows: 294891
            })}, '/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp/train_corpus_ko2km_274479.tsv': {'ko_KR2km_KH': Dataset({
                features: ['translation'],
                num_rows: 291978
            })}, '/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp/train_corpus_ko2vi_274479.tsv': {'ko_KR2vi_VN': Dataset({
                features: ['translation'],
                num_rows: 304959
            })}}
        """
        hf_dataset_bundle = {corpus_path : '' for corpus_path in corpus_paths}
        
        for corpus_path in corpus_paths:
            corpus = pd.read_csv(corpus_path, sep="\t")

            #lang_pair = [lang for lang in self.LANGS if lang in corpus.columns]
            lang_pair = [lang for lang in corpus.columns[2:] if lang in self.LANGS]
            #ex) ["en_XX", "ko_KR"]
            try:
                assert len(lang_pair) == 2
            except:
                raise ValueError(f"Check the if ID of LANGS is validate")
            src_lang, tgt_lang = lang_pair

            assert len(src_lang) == 5
            assert len(tgt_lang) == 5

            if src_lang not in self.LANGS and tgt_lang not in self.LANGS:
                raise ValueError(f"The support languages are {self.LANGS}")
        
            self.logger.info(f"\n{corpus.head()}")
            src_data, tgt_data = corpus[src_lang], corpus[tgt_lang]

            if packing and (packing_size is not None):
                self.logger.info("Merge sentences into Segments...")
                packed_src, packed_tgt, packed_len, time_num = packing_data(
                    self.tokenizer, corpus, group_key, src_lang, tgt_lang, packing_size, self.preprocessor.properties["max_token_length"], merge_direction="bidirection"
                )
                
                if hybrid:
                    self.logger.info("Hybrid option : Prepare train data using sents & segments unit")
                    # prepare segments unit from packed data
                    packed_src_data, packed_tgt_data = [], []
                    packed_src_data.extend(shuffle_packed_data(seed=0, packed_data=packed_src))
                    packed_tgt_data.extend(shuffle_packed_data(seed=0, packed_data=packed_tgt))

                    assert len(src_data) == len(tgt_data)
                    assert len(packed_src_data) == len(packed_tgt_data)
                    
                    self.logger.info(f"len of packed src({src_lang}) & tgt({tgt_lang}) data : {len(packed_src_data)}")
                    self.logger.info(f"{src_lang} and {tgt_lang} data len : {len(src_data)}")

                    src_data = [s.strip() for s in src_data]
                    tgt_data = [s.strip() for s in tgt_data]
                    src_data.extend(["\n".join(sents) for sents in packed_src_data])
                    tgt_data.extend(["\n".join(sents) for sents in packed_tgt_data])
                    
                    random.seed(10)
                    ixs = list(range(len(src_data)))
                    random.shuffle(ixs)
                    src_data = [src_data[i] for i in ixs]
                    tgt_data = [tgt_data[i] for i in ixs]

                    self.logger.info(f"total number of parallel data({src_lang} > {tgt_lang}) : {len(src_data)}")

                    # check the pair data
                    sample = random.sample(range(len(src_data)), 2)
                    for i in sample:
                        self.logger.info("*" * 100)
                        self.logger.info(f"Sample Source Sentence({src_lang},#{i}) :{src_data[i]}")
                        self.logger.info(f"Sample Target Sentence({tgt_lang},#{i}) :{tgt_data[i]}")

                else:
                    self.logger.info("Prepare train data using only segments pairs(not use sentence pairs")
                    src_data, tgt_data = ["\n".join(sents) for sents in packed_src], ["\n".join(sents) for sents in packed_tgt]
            else:
                self.logger.info("Not use to packed(segments) data..Only use sentence pairs")
                src_data, tgt_data = corpus[src_lang], corpus[tgt_lang]

            hf_dataset_bundle[corpus_path] = {
                                                f"{src_lang}2{tgt_lang}" :
                                                    Dataset.from_dict(
                                                                    {
                                                                        "translation": [
                                                                                            {
                                                                                                src_lang: src.rstrip("\n"),
                                                                                                tgt_lang: tgt.rstrip("\n"),
                                                                                            }
                                                                                                for src, tgt in tqdm(zip(src_data, tgt_data), total=len(src_data))
                                                                                        ]
                                                                    }
                                                                )
                                              } 
                                                            
        return hf_dataset_bundle

    def get_tokenized_dataset(self, batch_size=20000, num_proc=8):
        '''
        self.tokenized_datasets : example data is below. 
                DatasetDict({
                    train: Dataset({                                                                            
                        features: ['input_ids', 'attention_mask', 'labels'],
                        num_rows: 2748854
                    })
                    valid: Dataset({
                        features: ['input_ids', 'attention_mask', 'labels'],
                        num_rows: 2956
                    })
                }
        '''
        self.tokenized_datasets = datasets.DatasetDict()
        src2tgt_bundle = list(self.raw_datasets.keys())

        for src2tgt in src2tgt_bundle:
            self.preprocessor.properties["src_lang"] = src2tgt.split('2')[0]
            self.preprocessor.properties["tgt_lang"] = src2tgt.split('2')[-1]

            self.logger.info(f"src2tgt to process : {src2tgt}")

            processed_dataset = self.raw_datasets[src2tgt].map(
                                                                self.preprocessor.preprocess,
                                                                batched=True,
                                                                batch_size=batch_size,
                                                                remove_columns=self.raw_datasets[src2tgt]["train"].column_names,
                                                                fn_kwargs=self.preprocessor.properties,
                                                                num_proc=num_proc
                                                                )

            if len(self.tokenized_datasets) == 0:
                self.tokenized_datasets.update(processed_dataset)
            else:
                for split_column in self.tokenized_datasets:
                    self.tokenized_datasets[split_column] = concatenate_datasets([self.tokenized_datasets[split_column], processed_dataset[split_column]])

        return self.tokenized_datasets


class Processor:
    def __init__(self, mbart_tokenizer, max_token_length=512, drop=True, bi_direction=True):
        """
        Arguments for huggingface dataset's user defined map function
        Ref) During pre-training, mbart use instance format of up to 512 tokens
        """
        self.properties = {
            "mbart_tokenizer": mbart_tokenizer,
            "max_token_length": max_token_length,
            "src_lang":"",
            "tgt_lang":"",
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
