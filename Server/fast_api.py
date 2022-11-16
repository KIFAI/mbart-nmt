import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import itertools

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Translate.inference import Translator

par_dir = os.path.dirname(os.path.abspath(os.path.dirname('utils.py')))
model_path = os.path.join(par_dir, 'FastModel/Ctrans-MBart/ctrans_fp16')
translator = Translator(model_path=model_path, model_type='Ctranslate2', device='cuda', device_index=[2,3],
                        max_length=200, batch_size=8)

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
    q: str
    source: str
    target: str

@app.post("/")
async def translate(item: Item):
    req = dict(item)
    print(req)

    start = time.time()
    results = translator.generate(req['q'], src_lang=req['source'], tgt_lang=req['target'])
    hypotheses = list(itertools.chain(*results))
    end = time.time()

    print(f"\n**pred**\n{' '.join(hypotheses)}")
    print(f"Elaped time : {end-start}\n")

    return {'translatedText' : ' '.join(hypotheses)}
