import os
import pprint
import itertools
import numpy as np
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_data(data_path=None):
    if data_path is None:
        en_sents = [['In June 2020, the week after a police officer murdered George Floyd, Jeff Bezos endorsed Black Lives Matter.', '“I support this movement,” Bezos replied via email to a customer who’d complained about seeing a BLM banner on Amazon.com.', 'A couple of days later, when Bezos received a fresh complaint that included a racist slur, he raised his commitment.', '“You’re the kind of customer I’m happy to lose,” he said.', 'He posted the exchanges on Instagram, where each drew hundreds of thousands of likes.', 'So it was strange that, at about the same time, store managers at Amazon-owned Whole Foods Market began telling workers not to wear clothes with BLM slogans, and punishing those who did.'], ['When President Ronald Reagan began his first term in 1981, the US economy was struggling. Unemployment rates were high and getting higher, and in 1979, inflation had peaked at an all-time high for peacetime.', "In an effort to combat these issues, Reagan's administration introduced a number of economic policies, including tax cuts for large corporations and high-income earners. The idea was that tax savings for the rich would cause extra money to trickle down to everyone else, and for that reason, these policies are often referred to as trickle-down economics.", "From the 80s to the late 90s, the US saw one of its longest and strongest periods of economic growth in history. Median income rose, as did rates of job creation."]]

        ko_sents = [["2020년 6월, 경찰관이 조지 플로이드를 살해한 다음 주, 제프 베이조스는 Black Lives Matter를 지지했다.", '베조스는 Amazon.com에서 BLM 배너를 본 것에 대해 불평한 고객에게 이메일을 통해 "나는 이 움직임을 지지한다"고 답했다.', "며칠 후, 베조스는 인종차별적 비방이 포함된 새로운 불평을 받았을 때, 그의 공약을 제기했다.", '“당신은 내가 잃어버려도 기쁜 고객입니다.”라고 그는 말했습니다.', "그는 인스타그램에 그 대화들을 올렸고, 각각의 대화들은 수십만개의 좋아요를 받았다.", "그래서 거의 동시에 아마존 소유의 홀 푸드 마켓의 매장 관리자들이 노동자들에게 BLM 슬로건이 적힌 옷을 입지 말라고 말하고, 입었던 사람들을 처벌하기 시작한 것은 이상했다."], ["1981년 레이건 대통령이 첫 임기를 시작했을 때, 미국 경제는 어려움을 겪고 있었습니다. 이미 높았던 실업률은 더욱 더 상승했고 1979년에는 물가 상승률이 전시를 제외하고 사상 최고였습니다.","이런 문제들을 해결하기 위해 레이건 행정부는 여러 가지 경제 정책을 도입했는데 대기업과 고소득층에 대한 감세도 들어 있었습니다. 고소득층에 대한 감세가 다른 모든 사람들에게로 흘러내려 분배된다는 이론에서 비롯되었고 그 이유 때문에 이 정책들은 흔히 낙수 경제라고 불렸습니다.","80년대에서 90년대 후반까지 미국은 역사상 가장 길고 강력한 경제적 성장기를 맞이했습니다. 중산층의 소득이 증가하고 일자리도 늘어났습니다."]]

    else:
        with open(f"{data_path}/test.en_XX", 'r') as en, open(f"{data_path}/test.ko_KR") as ko:
            en_sents, ko_sents = en.readlines(), ko.readlines()
            en_sents, ko_sents = [[s.strip()] for s in en_sents], [[s.strip()] for s in ko_sents]

    print(f"Test sets # {len(en_sents)}")
        
    return {'en_XX' : en_sents, 'ko_KR' : ko_sents}

def load_model(plm_path, device):
    model = MBartForConditionalGeneration.from_pretrained(plm_path).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(plm_path)
    
    return model, tokenizer

def generate(model, tokenizer, src_sents, src_lang, tgt_lang, batch_size=1, device='cpu'):
    tokenizer.src_lang = src_lang

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    sentence_batch = batch(src_sents, batch_size)
    results = []
    for src_sents in sentence_batch:
        encoded = tokenizer(src_sents, padding=True, return_tensors='pt').to(device)
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], num_beams=2)
        results.append(tokenizer.batch_decode(generated, skip_special_tokens=True))
    
    return list(itertools.chain(*results))

def evaluate(tokenizer, preds, refs):
    bleu_unigram = []
    bleu_ngram = []
    bleu_total = []
    for hyp, ref in zip(preds, refs):
        score_unigram = sentence_bleu([tokenizer.tokenize(ref)], tokenizer.tokenize(hyp), weights=(1,0,0,0), smoothing_function=SmoothingFunction().method4)
        bleu_unigram.append(score_unigram)
        score_ngram = sentence_bleu([tokenizer.tokenize(ref)], tokenizer.tokenize(hyp), weights=(0,0.34,0.33,0.33), smoothing_function=SmoothingFunction().method4)
        bleu_ngram.append(score_ngram)
        score_total = sentence_bleu([tokenizer.tokenize(ref)], tokenizer.tokenize(hyp), weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method4)
        bleu_total.append(score_total)
    return {'bleu_unigram' : np.mean(bleu_unigram), 'bleu_ngram' : np.mean(bleu_ngram), 'bleu_total' : np.mean(bleu_total)}

def save_trans_hyps(save_path, sents_list):
    with open(save_path, 'w') as f:
        f.write('\n'.join(sents_list))
    

def main(model_paths, data=None, src_lang='ko_KR', tgt_lang='en_XX', 
        reverse=True, doc_nmt=True, device='cpu'):

    if reverse:
        src_lang, tgt_lang = tgt_lang, src_lang
    else:
        pass

    trans_type = 'doc' if doc_nmt else 'sent'

    if doc_nmt:
        print(f'Translate from {src_lang} to {tgt_lang} by segments unit\n')
        src_sents, tgt_sents = [' '.join(d) for d in data[src_lang]], [' '.join(d) for d in data[tgt_lang]]
    else:
        print(f'Translate from {src_lang} to {tgt_lang} by sentence unit\n')
        src_sents, tgt_sents = list(itertools.chain(*data[src_lang])), list(itertools.chain(*data[tgt_lang]))
    
    assert isinstance(model_paths, list) == True
    for path in model_paths:
        if os.path.exists(path):
            print(f"Version : {path.split('final_checkpoint_')[-1]}")
            print('trans_type', trans_type)
            model, tokenizer = load_model(path, device)
            result = generate(model, tokenizer, src_sents, src_lang, tgt_lang, batch_size=16, device=device)
            save_trans_hyps(f"./src/hypothesis/{path.split('final_checkpoint_')[-1]}_from_{src_lang}_to_{tgt_lang}_{trans_type}", result)
            bleus = evaluate(tokenizer, preds=result, refs=tgt_sents)

            pprint.pprint(bleus)


#data = load_data(data_path='./src/train_corpus/cased_corpus_v2')
data = load_data()

model_paths = ['./src/ftm/cased_mbart50_v3-finetuned-en_XX-to-ko_KR/final_checkpoint_sent_unit',
        './src/ftm/cased_mbart50_v3-finetuned-en_XX-to-ko_KR/final_checkpoint_seg_unit',
        './src/ftm/cased_mbart50_v3-finetuned-en_XX-to-ko_KR/final_checkpoint_hybrid',
        './src/ftm/cased_mbart50_v3-finetuned-en_XX-to-ko_KR/final_checkpoint_multiple_batch']

trans_directions = [[False, False], [True, False], [False, True], [True, True]]
#trans_directions = [[False, False], [True, False]]

for r, d in trans_directions:
    main(model_paths, data=data, src_lang='ko_KR', tgt_lang='en_XX', 
            reverse=r, doc_nmt=d, device='cuda:3')
    print('*****************************************')

