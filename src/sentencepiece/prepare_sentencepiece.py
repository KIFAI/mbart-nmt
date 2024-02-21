import re, os, glob, shutil
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from transformers import MBart50TokenizerFast

TGT_LANG = ['vi', 'id', 'km', 'hi']
EXCLUDE_LANG = 'en'
EXCLUDE_PATTERN = re.compile("▁?[\uAC00-\uD7AF|\u1100-\u11FF|\uA960-\uA97F|\uD7B0-\uD7FF|\u3130-\u318F]+") 
LANG2LID = {'ko':'ko_KR', 'vi':'vi_VN', 'id':'id_ID', 'km':'km_KH', 'hi':'hi_IN'}

corpus = []

BASE_DIR = '/opt/project/mbart-nmt/src'

TRAIN_CORPUS_DIR = [f for f in glob.glob(os.path.join(BASE_DIR, 'train_corpus/cased_corpus_exp/train*')) if EXCLUDE_LANG not in f]
print(f"TRAIN CORPUS : {TRAIN_CORPUS_DIR}")
VALID_CORPUS_DIR = [f for f in glob.glob(os.path.join(BASE_DIR, 'train_corpus/cased_corpus_exp/valid*')) if EXCLUDE_LANG not in f]
print(f"VALID CORPUS : {VALID_CORPUS_DIR}")

SAVE_DIR, SAVE_FN = os.path.join(BASE_DIR, 'train_corpus/spc_corpus'), f"{'_'.join(TGT_LANG)}_corpus.out"

for f in TRAIN_CORPUS_DIR:
    corpus_df = pd.read_csv(f, sep="\t")

    regex = re.compile(r'([a-zA-Z]+2[a-zA-Z]+)')
    match = regex.search(f)
    src_col, tgt_col = LANG2LID[match.group().split('2')[0]], LANG2LID[match.group().split('2')[-1]]
    print(f"src col : {src_col}, tgt col : {tgt_col}")

    for LANG in TGT_LANG :
        if LANG2LID[LANG] in [src_col, tgt_col]:
            print(f"The corpus language to train : {LANG2LID[LANG]}")
            corpus.extend(corpus_df[LANG2LID[LANG]].to_list())

print(f"\nShape of corpus merged : {len(corpus)}")
corpus = [s for s in corpus if EXCLUDE_PATTERN.search(s) is None]
print(f"Shape of corpus merged after processing EXCLUDE PATTERN: {len(corpus)}\n")
corpus = list(set(corpus))
print(f"Shape of corpus merged after processing DUPLICATE CASE: {len(corpus)}\n")

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

with open(os.path.join(SAVE_DIR, SAVE_FN), 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(list(set(corpus))))

INPUT, SPC_DIR, TMP_DIR = os.path.join(SAVE_DIR, SAVE_FN), os.path.join(BASE_DIR, f"sentencepiece/{'_'.join(TGT_LANG)}"), os.path.join(BASE_DIR, 'sentencepiece/tmp')
PREFIX, CHR_COVERAGE, VOCAB_SIZE = os.path.join(SPC_DIR, '_'.join(TGT_LANG)), 1.0, 10000

if not os.path.isdir(SPC_DIR):
    os.mkdir(SPC_DIR)
    print(f"Make dir of sentencepiece : {SPC_DIR}")

print(f"INPUT : {INPUT}\nOUTPUT_DIR : {SPC_DIR}\nPREFIX: {PREFIX.split('/')[-1]}\nCHR_COVERAGE: {CHR_COVERAGE}\nVOCAB_SIZE: {VOCAB_SIZE}")

tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
tokenizer.save_pretrained(TMP_DIR)
os.remove(f"{TMP_DIR}/sentencepiece.bpe.model")
os.remove(f"{TMP_DIR}/tokenizer.json")

spm.SentencePieceTrainer.train(
    f"--input={INPUT} --model_prefix={PREFIX} --character_coverage={CHR_COVERAGE} --vocab_size={VOCAB_SIZE}" +
    " --vocabulary_output_piece_score=true" +
    " --model_type=bpe")

shutil.copy(f"{PREFIX}.model", f"{TMP_DIR}/sentencepiece.bpe.model")

tokenizer = MBart50TokenizerFast.from_pretrained(TMP_DIR)
tokenizer.save_pretrained(SPC_DIR)
print(f"New vocab size is {tokenizer.vocab_size}")

for f in VALID_CORPUS_DIR:
    corpus_df = pd.read_csv(f, sep="\t")

    regex = re.compile(r'([a-zA-Z]+2[a-zA-Z]+)')
    match = regex.search(f)
    src_col, tgt_col = LANG2LID[match.group().split('2')[0]], LANG2LID[match.group().split('2')[-1]]
    print(f"\nsrc col : {src_col}, tgt col : {tgt_col}")

    input_langs = []
    for LID in [src_col, tgt_col]:
        if LID in [LANG2LID[L] for L in TGT_LANG]:
            input_langs.append(LID)
    print(f"Input Langs : {input_langs}")

    if len(input_langs) != 0 :
        for input_lang in input_langs:
            print(f"The file name : {f}")
            print(corpus_df.head(3))
            data = corpus_df[input_lang].to_list()
            tokenizer.src_lang = input_lang
            print(f"SRC LANG of tokenizer is {tokenizer.src_lang}")

            input_ids = tokenizer(data).input_ids
            print(f"The Unknown Token ID is : {tokenizer.unk_token_id}")
            seq_with_unk = [seq for seq in input_ids if tokenizer.unk_token_id in seq]
            print(f"The number of unk token in {tokenizer.src_lang} data(#{len(data)}) : {len(seq_with_unk)}")
            print(f"{tokenizer.batch_decode(seq_with_unk)}\n")

