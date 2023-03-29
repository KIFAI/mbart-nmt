import json
import requests
import pandas as pd
from multiprocessing import Pool

def batch(iterable, n=1):
    '''
    Generator configuring a list of sentences by a predefined batch size
    '''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def divide(d):
    return d['score']/d['tgt_tok_len']

def translate_batch(batch_data):
    src_lang, tgt_lang, b = batch_data
    translated = requests.post('http://10.17.23.228:13001', headers={'Content-Type':'application/json'}, json={'q':b, 'source':src_lang, 'target':tgt_lang}).json()['translatedText']
    return list(map(divide, translated))

if __name__ == '__main__':
    sources = pd.read_csv('./train_corpus_sampled.tsv', sep='\t')
    length = sources.shape[0]
    sources = {'ko_KR':sources['ko_KR'].to_list(), 'en_XX': sources['en_XX'].to_list()}
    results = {}

    for src_lang, tgt_lang in [('ko_KR', 'en_XX'), ('en_XX', 'ko_KR')]:
        scores = []
        with Pool(16) as pool:
            batch_data = [(src_lang, tgt_lang, b) for b in batch(sources[src_lang], n=32)]
            scores = pool.map(translate_batch, batch_data)
            scores = [score for batch_scores in scores for score in batch_scores]
        results[src_lang] = scores

        with open('./src/scores_dist.json', 'w') as f:
            json.dump(results, f, indent=2)
