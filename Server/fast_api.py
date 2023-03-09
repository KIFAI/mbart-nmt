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
translator = Translator(model_path=model_path, model_type='Ctranslate2', device='cuda', device_index=[2,3],
                        max_length=240, batch_size=64)

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

@app.post("/")
async def translate(item: Item):
    req = dict(item)
    if isinstance(req['q'], str):
        req['q'] = [req['q']]
    start = time.time()
    hypotheses = translator.generate(req['q'], src_lang=req['source'], tgt_lang=req['target'])
    end = time.time()

    print(f"\n**pred**\n{hypotheses}")
    print(f"\nElapsed time for all process: {end-start}\n")

    return {'translatedText' : hypotheses}
