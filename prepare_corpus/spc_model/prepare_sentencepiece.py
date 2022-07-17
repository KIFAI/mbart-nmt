import os
import argparse
import sentencepiece as spm
from tqdm import tqdm
def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--corpus_path",
            default='/opt/translation/repo/mbart-nmt/src/pretokenized_aihub_corpus',
            type=str,
            )
    parser.add_argument(
            "--spc_path",
            default='/opt/translation/repo/mbart-nmt/src/sentencepiece',
            type=str,
            )

    parser.add_argument(
            "--input_fn",
            default="aihub_written_lower_pretok.mono",
            type=str,
            )

    parser.add_argument(
            "--out_fn",
            default="mono.lower_aihub",
            type=str,
            )

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':

    args = define_argparser()
    
    if os.path.exists(args.corpus_path):
        print(f'{args.corpus_path} path checked..')
        spm.SentencePieceTrainer.Train(f'--input={os.path.join(args.corpus_path, args.input_fn)} --model_prefix={os.path.join(args.spc_path, args.out_fn)} --vocab_size=51100 --vocabulary_output_piece_score=false --model_type=bpe')

        with open(f"{os.path.join(args.spc_path,args.out_fn)}.vocab", 'r', encoding='utf-8') as f:
            vocab = f.readlines()

        mbart_format_vocab = []

        for i in tqdm(range(len(vocab)), desc='Editing custom spc vocab into Mbart dict format....', total=len(vocab)):
            if vocab[i] in ["<unk>\n", "<s>\n", "</s>\n"]:
                print(f'Filter {vocab[i]} from {args.out_fn}.vocab')
                continue
            mbart_format_vocab.append(vocab[i].rstrip('\n') + " 1\n")
        
        with open(f'{os.path.join(args.spc_path, args.out_fn)}_mbart.vocab', 'w', encoding='utf-8') as f:
            f.write(''.join(mbart_format_vocab))
    else:
        raise Exception('Check sentencepiece path and corpus to train')
