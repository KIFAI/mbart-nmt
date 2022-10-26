import time
import itertools
import nltk
import ctranslate2

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (MBartForConditionalGeneration, MBart50TokenizerFast)

ctrans_path = "./Ctrans-MBart/ctrans_fp16"

tokenizer = MBart50TokenizerFast.from_pretrained(ctrans_path)
model = ctranslate2.Translator(ctrans_path, inter_threads=1, intra_threads=8, device="cuda", device_index=[4,5])

app = FastAPI()

origins = [
        "http://10.17.23.228:11757"
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def tokenize(x):
    return tokenizer.tokenize(x)

def convert_to_inputs(tokenized_sents, max_length=120):
        
    input_len, start_ix = 0, 0
    segments = []
    for i, t_s in enumerate(tokenized_sents):
        input_len += len(t_s)
        if i+1 == len(tokenized_sents):
            segments.append([tokenizer.src_lang] + list(itertools.chain(*tokenized_sents[start_ix:])) + [tokenizer.eos_token])
        elif input_len + len(tokenized_sents[i+1]) > max_length:
            segments.append([tokenizer.src_lang] + list(itertools.chain(*tokenized_sents[start_ix:i+1])) + [tokenizer.eos_token])
            input_len = 0
            start_ix = i+1
        else:
            pass
    return segments

def detokenize(x):
    return tokenizer.convert_tokens_to_string(x.hypotheses[0][1:]).replace('<unk>', '')

class Item(BaseModel):
    q: str
    source: str
    target: str

@app.post("/")
async def translate(item: Item):
    req = dict(item)
    print(req)
    start = time.time()
    tokenizer.src_lang = req["source"]

    splitted_sents = nltk.sent_tokenize(req['q'])
    print(f"Splitted len : {len(splitted_sents)}")
    if len(splitted_sents) == 1 :
        inputs = [[tokenizer.src_lang] + tokenizer.tokenize(splitted_sents[0]) + [tokenizer.eos_token]]
    else:
        inputs = convert_to_inputs(list(map(tokenizer.tokenize, splitted_sents)))
        print(f"Segments len : {len(inputs)}")
        #print('\n'.join(list(map(tokenizer.convert_tokens_to_string, inputs))))
    
    try:
        assert req['q'] == tokenizer.convert_tokens_to_string(list(itertools.chain(*[t[1:-1] for t in inputs])))
    except Exception as ex:
        print('************')
        print(tokenizer.convert_tokens_to_string(list(itertools.chain(*[t[1:-1] for t in inputs]))))
        print('************')
    
    translated_tokens = model.translate_batch(source=inputs, target_prefix=[[req["target"]]]*len(inputs) ,beam_size=2, asynchronous=False)
    pred = list(map(detokenize, translated_tokens))
    end = time.time()
    print(f"pred: {pred}\nElaped time : {end-start}")
    return {'translatedText' : ' '.join(pred)}
