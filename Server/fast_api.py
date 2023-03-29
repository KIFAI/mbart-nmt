import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time

from typing import List, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Translate.inference import Translator

par_dir = os.path.dirname(os.path.abspath('./'))
model_path = os.path.join(par_dir, 'FastModel/Ctrans-MBart/ctrans_fp16')
translator = Translator(model_path=model_path, model_type='Ctranslate2', device='cuda', device_index=[1,2,3],
                        max_length=240, batch_size=32)

app = FastAPI()

origins = [
        "http://10.17.23.228:14001"
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    q: Union[str, List[str]]
    source: str
    target: str

def process_multi_lines(translated_multi_lines):
        result = {}
        for d in translated_multi_lines:
            for key, value in d.items():
                if key == 'translated':
                    if key not in result:
                        result[key] = value
                    else:
                        result[key] += f"\n{value}"
                else:
                    if key not in result:
                        result[key] = value
                    else:
                        result[key] += value
        return result

@app.post("/")
async def translate(item: Item):
    req = dict(item)
    if isinstance(req['q'], str):
        req['q'] = [req['q']]

    query_order = ['single' if len(q.split('\n')) == 1 else 'multi' for q in req['q']]
    single_lines = [req['q'][i].strip() for i, s in enumerate(query_order) if query_order[i] == 'single']
    multi_lines = [req['q'][i].strip().split('\n') for i, s in enumerate(query_order) if query_order[i] == 'multi']
    
    if list(set(query_order)) == ['single']:
        start = time.time()
        translated_single_lines = translator.generate(single_lines, src_lang=req['source'], tgt_lang=req['target'])
        end = time.time()

        print(f"\n**pred**\n{translated_single_lines}")
        print(f"\nElapsed time for translating query with single lines : {end-start}\n")

        return {'translatedText' : translated_single_lines}

    elif list(set(query_order)) == ['multi']:
        start = time.time()
        translated_multi_lines = [process_multi_lines(translator.generate(m_l, src_lang=req['source'], tgt_lang=req['target'])) for m_l in multi_lines]
        end = time.time()

        print(f"\n**pred**\n{translated_multi_lines}")
        print(f"\nElapsed time for translating query with multi lines : {end-start}\n")

        return {'translatedText' : translated_multi_lines}

    else:
        start = time.time()
        translated_single_lines = translator.generate(single_lines, src_lang=req['source'], tgt_lang=req['target'])
        translated_multi_lines = [process_multi_lines(translator.generate(m_l, src_lang=req['source'], tgt_lang=req['target'])) for m_l in multi_lines]
        end = time.time()

        hypotheses, single_index = [], 0
        for sent_type in query_order:
            if sent_type == 'single':
                hypotheses.append(translated_single_lines[single_index])
                single_index += 1
            else:
                hypotheses.append(translated_multi_lines.pop(0))

        print(f"\n**preds of single and multi lines**\n{hypotheses}")
        print(f"\nElapsed time for translating query with single and multi lines : {end-start}\n")

        return {'translatedText' : hypotheses}

