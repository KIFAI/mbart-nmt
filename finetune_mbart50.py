import argparse
import glob
from data_loader.nmt_loader import NmtDataLoader, Processor
from transformers import (EarlyStoppingCallback, MBartForConditionalGeneration, MBart50TokenizerFast,
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
        default="./src/plm/reduced_hf_mbart50_m2m",
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
        "--max_token_length",
        default=256,
        type=int
    )
    parser.add_argument(
        '--drop_case',
        action='store_true',
        help='Whether to drop sequence if it is longer than max_token length',
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=4,
        type=int,
        help='When the batch size is small, it can be used if you want to have the same effect as the batch size as many as a parameter multiple.'
    )
    parser.add_argument(
        "--num_epochs",
        default=3,
        type=int
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to use 16-bit (mixed) precision training (through NVIDIA Apex) instead of 32-bit training.',
    )
    parser.add_argument(
        '--bi_direction',
        action='store_true',
        help='Whether to train bi-direcional NMT Engine instead of uni_directional training.',
    )
    parser.add_argument(
        '--packing_data',
        action='store_true',
        help='Merge sentences into segments',
    )
    parser.add_argument(
        "--packing_size",
        default=256,
        type=int
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Prepare train data using sents & segments unit',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training pre-checkpoint',
    )

    args = parser.parse_args()

    return args

def main(args):
    mbart_plm = MBartForConditionalGeneration.from_pretrained(args.plm)
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(args.plm)

    preprocessor = Processor(mbart_tokenizer, args.src_lang, args.tgt_lang, args.max_token_length,
            args.drop_case, args.bi_direction)
    loader = NmtDataLoader(mbart_tokenizer, preprocessor, args.corpus_path, args.packing_data, args.packing_size, args.hybrid)
    tokenized_datasets = loader.get_tokenized_dataset(batch_size=20000, num_proc=8)
    print('\n', tokenized_datasets, '\n')

    data_collator = DataCollatorForSeq2Seq(mbart_tokenizer, model=mbart_plm, padding='longest')
    batch = data_collator([tokenized_datasets["train"][i] for i in range(0, 2)])
    print(f"batch_keys : {batch.keys()}")
    print(f"target_labels : {batch['labels']}")
    print(f"decoder_input ids : {batch['decoder_input_ids']}")
    print(f"decoded inputs : {mbart_tokenizer.batch_decode(batch['decoder_input_ids'])}")
    print(f"SRC LANG & ID > {args.src_lang} : {mbart_tokenizer.vocab[args.src_lang]}")
    print(f"TGT LANG & ID > {args.tgt_lang} : {mbart_tokenizer.vocab[args.tgt_lang]}")
    print(f"mBart's EOS Token & ID > {mbart_tokenizer.eos_token} : {mbart_tokenizer.eos_token_id}\n")
    
    output_dir = f"{args.base_path}/src/ftm/{args.exp_name}-finetuned-{args.src_lang}-to-{args.tgt_lang}"
    train_args = Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,  # default 5e-5 > 3e-5
        adam_beta1=0.9,
        adam_beta2=0.999,  # default 0.999 > 0.98
        adam_epsilon=1e-08,  # default 1e-8 > 1e-06
        lr_scheduler_type='linear',  # default 'linear' > 'polynomial'
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.0,  # default 0.0
        label_smoothing_factor=0.0,  # default 0.0
        save_total_limit=1,
        save_steps=500,
        #load_best_model_at_end=True,
        logging_steps=500,
        logging_strategy='steps',
        log_level='info',
        report_to='tensorboard',
        num_train_epochs=args.num_epochs,
        fp16=True
    )
    print(f"Tensorboard logging dir : {train_args.logging_dir}")

    trainer = Seq2SeqTrainer(
        mbart_plm,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=mbart_tokenizer,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        # compute_metrics=compute_metrics
    )
    if args.resume:
        ckpts = glob.glob(f'{args.base_path}/src/ftm/{args.exp_name}-finetuned-{args.src_lang}-to-{args.tgt_lang}/checkpoint-*')
        resume_latest_ckpt = sorted(ckpts)[-1]
        print(f'Resume training {resume_latest_ckpt}')
        trainer.train(
                resume_from_checkpoint=resume_latest_ckpt)
    else:
        trainer.train()

    trainer.save_model(
        f'{args.base_path}/src/ftm/{args.exp_name}-finetuned-{args.src_lang}-to-{args.tgt_lang}/final_checkpoint')


if __name__ == '__main__':
    args = define_argparser()
    print(args)
    main(args)
