import os
import time
import argparse
import sentencepiece as spm
from transformers import (MBartForConditionalGeneration, MBartTokenizer)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, default='mbart-fp16')
    p.add_argument("--src_lang", default="en_XX", type=str)
    p.add_argument("--tgt_lang",default="ko_KR",type=str)
    p.add_argument('--base_path', type=str, default='/opt/project/translation/mbart-nmt/src')
    p.add_argument('--src_path', type=str, default='raw_corpus/data_with_upper_lc/test.en_XX')
    p.add_argument('--tgt_path', type=str, default='raw_corpus/data_with_upper_lc/test.ko_KR')
    p.add_argument('--sample_n', type=int, default=10)
    p.add_argument('--check_sample', default=False, action='store_true')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--fp16', default=False, action='store_true')
    p.add_argument('--cuda_id', type=int, default=0)
    config = p.parse_args()

    return config

def load_data(base_path, file_path, n = 10):
    with open(os.path.join(base_path, file_path)) as f:
        lines = f.readlines()[:n]
    return [s.strip() for s in lines]

def write_data(base_path, hyp_path, data):
    saved_path = os.path.join(base_path, hyp_path)

    with open(saved_path, 'w') as f:
        f.write('\n'.join(data))

def generate(model, ftm_path, sentence_bucket, batch_size = 3, cuda_id = 0, half_precision=True):    
    if half_precision:
        model.half()
    tokenizer = MBartTokenizer.from_pretrained(ftm_path)

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    sentence_batch = batch(sentence_bucket, batch_size)
    hyps = []
    start_time = time.time()

    for sents in sentence_batch:
        inputs = tokenizer(sents, return_tensors="pt", padding=True).to(f'cuda:{cuda_id}')
        translated_tokens = model.generate(
            **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], num_beams=3, early_stopping=True, max_length=150)
        pred = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)
        hyps.extend(pred)

    end_time = time.time()
    print(f'Elapsed time : {end_time-start_time}')
    
    return hyps

if __name__ == '__main__':
    config = define_argparser()
    srcs = load_data(config.base_path, config.src_path, config.sample_n)
    refs = load_data(config.base_path, config.tgt_path, config.sample_n)

    model_path = f"{config.base_path}/ftm/{config.exp_name}-finetuned-{config.src_lang}-to-{config.tgt_lang}/final_checkpoint"
    ft_model = MBartForConditionalGeneration.from_pretrained(model_path).to(f"cuda:{config.cuda_id}")
    hyps = generate(ft_model, model_path, srcs, config.batch_size, config.cuda_id, half_precision=config.fp16)
    write_data(config.base_path, f'hypothesis/{config.exp_name}_lc_hypothesis', hyps)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_path}/sentencepiece.bpe.model")

    #Evaluation
    bleu_unigram = []
    bleu_ngram = []
    for src,ref,hyp in zip(srcs, refs, hyps):
        if config.check_sample:
            print(f"srcs:{src}\nref:{ref}\nhyp:{hyp}\n")
        score_unigram = sentence_bleu([sp.encode_as_pieces(ref)], sp.encode_as_pieces(hyp), weights=(1,0,0,0), smoothing_function=SmoothingFunction().method4)
        bleu_unigram.append(score_unigram)
        score_ngram = sentence_bleu([sp.encode_as_pieces(ref)], sp.encode_as_pieces(hyp), weights=(0,0.34,0.33,0.33), smoothing_function=SmoothingFunction().method4)
        bleu_ngram.append(score_ngram)
    print(f"EXP NAME : {config.exp_name} Bleu-UNIGRAM Score : {sum(bleu_unigram)/len(bleu_unigram)}")
    print(f"EXP NAME : {config.exp_name} Bleu-NGRAM Score : {sum(bleu_ngram)/len(bleu_ngram)}")

    tokens_nums = [(len(sp.encode_as_pieces(ref)), len(sp.encode_as_pieces(hyp))) for ref, hyp in zip(refs, hyps)]
    ref_avg_tokens = sum([r for r, h in tokens_nums])/len(refs)
    hyp_avg_tokens = sum([h for r, h in tokens_nums])/len(hyps)
    ratio_hyp_to_ref = sum([h/r for r, h in tokens_nums])/len(tokens_nums)
    print(
        f'Total sentence : {len(refs)}\nAvg ref tokens : {ref_avg_tokens}\nAvg hyp tokens : {hyp_avg_tokens}\nratio of len(hyp/ref) : {ratio_hyp_to_ref}')
