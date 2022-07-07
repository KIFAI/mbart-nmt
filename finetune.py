from transformers import MBartForConditionalGeneration
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import datasets

from tqdm import tqdm
from datasets import Dataset, load_metric
from transformers import (MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, set_seed)

# Helpful function for reproducible training from setting the seed in random, numpy, torch, and/or tf
set_seed(0)


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default="/opt/project/translation/repo/mbart-nmt",
        type=str,
    )
    parser.add_argument(
        "--corpus_path",
        default="./src/raw_corpus/data_with_upper_lc",
        type=str,
    )
    parser.add_argument(
        "--plm",
        default="./src/plm/reduced_mbart.cc25",
        type=str,
    )
    parser.add_argument(
        "--additional_special_tokens",
        default=None
    )
    parser.add_argument(
        "--src_lang",
        default="en_XX",
        type=str
    )
    parser.add_argument(
        "--tgt_lang",
        default="ko_KR",
        type=str,
    )
    parser.add_argument(
        "--exp_name",
        default='mbart-fp16',
        type=str,
    )
    parser.add_argument(
        "--max_token_lengh",
        default=150,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to use 16-bit (mixed) precision training (through NVIDIA Apex) instead of 32-bit training.',
    )

    args = parser.parse_args()

    return args


def get_parallel_dataset(base_path, src_lang='en_XX', tgt_lang='ko_KR', category='train'):
    '''
    Load splited src&tgt lang's corpus into huggingface dataset format
    '''
    category_data = []
    src_path = os.path.join(base_path, category)
    tgt_path = os.path.join(base_path, category)
    with open(f"{src_path}.{src_lang}", "r") as src, open(f"{tgt_path}.{tgt_lang}", "r") as tgt:
        src_data = src.readlines()
        tgt_data = tgt.readlines()
    for i, lines in enumerate(tqdm(zip(src_data, tgt_data), total=len(src_data))):
        category_data.append(
            {
                "translation": {
                    f"{src_lang}": lines[0].rstrip('\n'),
                    f"{tgt_lang}": lines[1].rstrip('\n'),
                }
            }
        )
    return Dataset.from_pandas(pd.DataFrame(category_data))


def compute_metrics(eval_preds, tokenizer):
    '''
    metirc function
    '''
    metric = load_metric("sacrebleu")
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    return {"bleu": result["score"]}


class Processor:
    def __init__(self, mbart_tokenizer, prefix='', max_token_length=150, src_lang='en_XX', tgt_lang='ko_KR'):
        '''
        Argments for huggingface dataset's user defined map function
        '''
        self.properties = {"mbart_tokenizer": mbart_tokenizer,
                           "prefix": prefix,
                           "max_token_length": max_token_length,
                           "src_lang": src_lang,
                           "tgt_lang": tgt_lang}

    @staticmethod
    def preprocess(examples, mbart_tokenizer, prefix, max_token_length, src_lang, tgt_lang):
        '''
        user defined map function for huggingface dataset 
        '''
        inputs = [prefix + ex[src_lang]
                  for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        model_inputs = mbart_tokenizer(
            inputs, max_length=max_token_length, truncation=True)

        # Setup the tokenizer for targets
        with mbart_tokenizer.as_target_tokenizer():
            labels = mbart_tokenizer(
                targets, max_length=max_token_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


if __name__ == '__main__':
    args = define_argparser()
    print(args)
    train_dataset = get_parallel_dataset(args.corpus_path, args.src_lang, args.tgt_lang, category='train')
    eval_dataset = get_parallel_dataset(args.corpus_path, args.src_lang, args.tgt_lang, category='valid')
    raw_datasets = datasets.DatasetDict(
        {"train": train_dataset, "validation": eval_dataset})
    
    mbart_plm = MBartForConditionalGeneration.from_pretrained(args.plm)

    if args.plm == 'facebook/mbart-large-50-one-to-many-mmt':
        mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
            args.plm, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    else:
        if args.additional_special_tokens is not None:
            if not isinstance(args.additional_special_tokens.split(','), list):
                raise TypeError('Check additional tokens arg format, "special token1, special_token2,..."')
            special_tokens=[f"<{st}>" for st in args.additional_special_tokens.split(',')]
            print(special_tokens)

            mbart_tokenizer = MBartTokenizer.from_pretrained(args.plm, src_lang=args.src_lang, tgt_lang=args.tgt_lang
                    , additional_special_tokens=special_tokens)
            mbart_plm.resize_token_embeddings(len(mbart_tokenizer))
            print(f"After vocab size : {mbart_plm.vocab_size}")
        else:
            mbart_tokenizer = MBartTokenizer.from_pretrained(args.plm, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    preprocessor = Processor(
        mbart_tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    tokenized_datasets = raw_datasets.map(
        preprocessor.preprocess, batched=True, remove_columns=raw_datasets["train"].column_names,
        fn_kwargs=preprocessor.properties)
    
    data_collator = DataCollatorForSeq2Seq(mbart_tokenizer, model=mbart_plm)
    batch = data_collator([tokenized_datasets["validation"][i]
                          for i in range(1, 3)])

    train_args = Seq2SeqTrainingArguments(
        f"{args.base_path}/src/ftm/{args.exp_name}-finetuned-{args.src_lang}-to-{args.tgt_lang}",
        evaluation_strategy="epoch",
        learning_rate=5e-5,  # default 5e-5 > 3e-5
        adam_beta1=0.9,
        adam_beta2=0.999,  # default 0.999 > 0.98
        adam_epsilon=1e-08,  # default 1e-8 > 1e-06
        lr_scheduler_type='linear',  # default 'linear' > 'polynomial'
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size/2,
        weight_decay=0.0,  # default 0.0
        label_smoothing_factor=0.0,  # default 0.0
        save_total_limit=1,
        save_steps=5000,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=args.fp16
    )

    trainer = Seq2SeqTrainer(
        mbart_plm,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=mbart_tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.train()

    trainer.save_model(
        f'{args.base_path}/src/ftm/{args.exp_name}-finetuned-{args.src_lang}-to-{args.tgt_lang}/final_checkpoint')
