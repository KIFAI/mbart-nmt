import argparse
import os
import sentencepiece as spm

parser = argparse.ArgumentParser()
'''
parser.add_argument(
        "--ko_file",
        default="/opt/project/translation/transformer-nmt/pretokenized_corpus/corpus_sample.train.tok.ko",
        type=str,
        )
parser.add_argument(
        "--en_file",
        default="/opt/project/translation/transformer-nmt/pretokenized_corpus/corpus_sample.train.tok.en",
        type=str,
        )
'''
parser.add_argument(
        "--mono_file",
        default="/opt/project/translation/repo/mbart-nmt/src/aihub_corpus/aihub_written.mono",
        type=str,
        )

parser.add_argument(
        "--out_fn",
        default="spc",
        type=str,
        )
args = parser.parse_args()
spm.SentencePieceTrainer.Train(f'--input={args.mono_file} --model_prefix=mono.{args.out_fn} --vocab_size=51100 --vocabulary_output_piece_score=false --model_type=bpe --max_sentence_length=9999')
#spm.SentencePieceTrainer.Train(f'--input={args.ko_file} --model_prefix=ko.{args.out_fn} --vocab_size=30000 --model_type=bpe --max_sentence_length=9999')
#spm.SentencePieceTrainer.Train(f'--input={args.en_file} --model_prefix=en.{args.out_fn} --vocab_size=50000 --model_type=bpe --max_sentence_length=9999')
