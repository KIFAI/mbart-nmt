import os,sys
import argparse
import datasets
import torch
import transformers

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import *
from data_loader.nmt_loader import NmtDataLoader, Processor

from datasets import load_dataset, load_from_disk
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, DataCollatorForSeq2Seq

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default="/home/jihyo/translation/repo/mbart-nmt",
        type=str,
    )
    parser.add_argument(
        "--tokenizer_path",
        default="./src/plm/reduced_hf_mbart50_m2m",
        type=str
    )
    parser.add_argument(
        "--corpus_path",
        default="./src/train_corpus/cased_corpus_exp_v2",
        type=str,
    )
    parser.add_argument(
        "--hf_dataset_path",
        default="./src/hf_dataset",
        type=str,
    )
    parser.add_argument(
        "--hf_dataset_name",
        default="custom_dataset",
        type=str,
    )
    parser.add_argument(
        "--num_proc",
        default=8,
        type=int
    )
    parser.add_argument(
        "--max_token_length",
        default=512,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=20000,
        type=int
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
        '--drop_case',
        action='store_true',
        help='Whether to drop sequence if it is longer than max_token length',
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
    args = parser.parse_args()

    return args


def main(args):
    """
    Creates a set of `Huggingface DatasetDict Object`s for the custom dataset,
    using "mBart50TokenizerFast" as the tokenizer.
    Args:
        args ('ArgumentParser'):
            'ArgumentParser' object
    """
    sys.path.append(args.base_path)
    print(sys.path)

    tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(args.base_path, args.tokenizer_path))

    preprocessor = Processor(tokenizer, args.max_token_length, args.drop_case, args.bi_direction)
    preparator = NmtDataLoader(tokenizer, preprocessor, args.corpus_path, args.packing_data, args.packing_size, args.hybrid)
    segment_datasets = preparator.get_tokenized_dataset(batch_size=args.batch_size, num_proc=args.num_proc)
    
    hf_dataset_dir = os.path.join(args.base_path, args.hf_dataset_path)
    if not os.path.isdir(hf_dataset_dir):
        os.mkdir(hf_dataset_dir)
        
    segment_datasets.save_to_disk(os.path.join(hf_dataset_dir, args.hf_dataset_name))
    
    assert segment_datasets.data == load_from_disk(os.path.join(hf_dataset_dir, args.hf_dataset_name)).data
    print(f"HF datasets are successfully prepared")

    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
