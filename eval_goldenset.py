import os
import evaluate, glob, itertools
import numpy as np
import pandas as pd
from Translate.inference import Translator

def evaluate_sacrebleu_metric(preds, refs, tgt_lang):
    metric = evaluate.load("sacrebleu")
    tokenizer = None if tgt_lang == 'en_XX' else 'ko-mecab'
    results = metric.compute(predictions=preds, references=[[ref] for ref in refs], tokenize=tokenizer)
    return results

if __name__ == '__main__':
    golden_fns = glob.glob('./src/golden_corpus/*.tsv')
    model_path = os.path.join(os.getcwd(), 'FastModel/Ctrans-MBart/ctrans_fp16')
    translator = Translator(model_path=model_path, model_type='Ctranslate2', device='cuda', device_index=[2,3],
                             max_length=200, batch_size=32)

    compare = False

    if compare:
        total_with_hyps = pd.read_csv('./src/golden_corpus/total_with_hyps.tsv', sep='\t')
        total_scores = {}
        total_scores['domain'] =  list(set(total_with_hyps['domain']))
        
        for src_lang, tgt_lang in [['en_XX', 'ko_KR'], ['ko_KR', 'en_XX']]:
            for name in ['papago', 'google', 'T5', 'kb']:
                col = f"{name}_{src_lang[:2]}2{tgt_lang[:2]}"
                print(col)

                domain_scores = []
                for domain in total_scores['domain']:
                    domain_df = total_with_hyps.loc[total_with_hyps['domain']==domain]
                    hyps, refs = domain_df[col].to_list(), domain_df[tgt_lang].to_list()
                    sacrebleu_score = evaluate_sacrebleu_metric(hyps, refs, tgt_lang)
                    domain_scores.append(round(sacrebleu_score['score'],3))
                total_scores[col] = domain_scores
                print(total_scores['domain'])
                print(domain_scores)
                
        eval_result_df = pd.DataFrame(total_scores)
        eval_result_df.to_csv('total_with_scores.tsv', sep='\t', index=False)
        print(eval_result_df)
                
    else:
        bleu_sacre = {}
        for src_lang, tgt_lang in [['en_XX', 'ko_KR'], ['ko_KR', 'en_XX']]:
            for fn in golden_fns:
                domain = fn.split('/')[-1].replace('.tsv', '')
                if domain not in ['은행', '스포츠', '예술', '금융경제', '의료', '기술과학', '식품농업', '문화관광', '법률', '뉴스', '교육강연', '정치', '대화']:
                    continue
                
                corpus_df = pd.read_csv(fn, sep='\t', index_col=0)
                data = {'en_XX' : corpus_df['en_XX'].to_list(), 'ko_KR' : corpus_df['ko_KR'].to_list()}
                print(f"Test sets of {domain} # : {corpus_df.shape[0]}")
                
                results = translator.generate(data[src_lang], src_lang=src_lang, tgt_lang=tgt_lang)
                hypotheses = list(itertools.chain(*results))

                #if f"kb{src_lang[:2]}2{tgt_lang[:2]}" not in corpus_df.columns:
                corpus_df[f"kb_{src_lang[:2]}2{tgt_lang[:2]}"] = hypotheses
                corpus_df.to_csv(f"./src/golden_corpus/{domain}.tsv", sep='\t', encoding='utf-8-sig')
                
                sacrebleu_score = evaluate_sacrebleu_metric(hypotheses, data[tgt_lang][:len(hypotheses)], tgt_lang)
                
                bleu_sacre[domain] = sacrebleu_score['score']
            print(f"{src_lang} -> {tgt_lang}\nTotal Sacre Bleu Score : {np.mean(list(bleu_sacre.values()))}")
            print(bleu_sacre)


