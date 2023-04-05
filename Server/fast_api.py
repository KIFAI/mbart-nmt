import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time

from typing import List, Union
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel
from Translate.inference import Translator
from Translate.docx_translator import trans_with_preserving_form

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

def translate_multi_lines(multi_lines, src_lang, tgt_lang):
    '''
    Args:
        multi_lines : splitted sentence by new line
    Returns:
        translated_multi_lines
    '''
    indices = []
    while '' in multi_lines:
        index = multi_lines.index('')
        indices.append(index)
        multi_lines.pop(index)
    
    translated_multi_lines = translator.generate(multi_lines, src_lang=src_lang, tgt_lang=tgt_lang)

    for index in reversed(indices):
        translated_multi_lines.insert(index, {"translated":'', "score":0, "tgt_tok_len":0, "src_chr_len":0, "src_tok_len":0})
    
    return translated_multi_lines

def process_multi_lines(translated_multi_lines):
    '''
    Args :
        translated_multi_lines : [{'translated' : sent1, 'score' : #, 'tgt_tok_len' : #, 'src_chr_len' : #, 'src_tok_len' : #}, {...}, ...}
    Returns :
        result
    '''
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
        if req['q'] == '':
            return {'translatedText' : [{"translated":'', "score":0, "tgt_tok_len":0, "src_chr_len":0, "src_tok_len":0}]}
        else:
            req['q'] = [req['q']]

    query_order = ['single' if len(q.split('\n')) == 1 and q.split('\n') != [''] else 'multi' for q in req['q']]
    single_lines = [line.strip() for line, order in zip(req['q'], query_order) if order == 'single']
    multi_lines = [line.strip().split('\n') for line, order in zip(req['q'], query_order) if order == 'multi']

    if set(query_order) == {'single'}:
        start = time.time()
        translated_single_lines = translator.generate(single_lines, src_lang=req['source'], tgt_lang=req['target'])
        end = time.time()

        #print(f"\n**pred**\n{translated_single_lines}")
        print(f"\nElapsed time for translating query with single lines : {end-start}\n")

        return {'translatedText' : translated_single_lines}

    elif set(query_order) == {'multi'}:
        start = time.time()
        translated_multi_lines = [process_multi_lines(translate_multi_lines(m_l, src_lang=req['source'], tgt_lang=req['target'])) for m_l in multi_lines]
        end = time.time()

        #print(f"\n**pred**\n{translated_multi_lines}")
        print(f"\nElapsed time for translating query with multi lines : {end-start}\n")

        return {'translatedText' : translated_multi_lines}

    else:
        start = time.time()
        translated_single_lines = translator.generate(single_lines, src_lang=req['source'], tgt_lang=req['target'])
        translated_multi_lines = [process_multi_lines(translate_multi_lines(m_l, src_lang=req['source'], tgt_lang=req['target'])) for m_l in multi_lines]
        end = time.time()
        
        hypotheses, single_index = [], 0
        for sent_type in query_order:
            if sent_type == 'single':
                hypotheses.append(translated_single_lines[single_index])
                single_index += 1
            else:
                hypotheses.append(translated_multi_lines.pop(0))
        
        #print(f"\n**preds of single and multi lines**\n{hypotheses}")
        print(f"\nElapsed time for translating query with single and multi lines : {end-start}\n")

        return {'translatedText' : hypotheses}

@app.post("/translate_docx")
async def translate_docx(docx_data:UploadFile, dest_lang:str):
    '''
    translate docx with preserving original docx
    Args:
        docx_data : only support microsoft word.docx
        dest_lang : target language that only support korean, english
    Returns:
        transformed_docx : translated docx with preserving origin docx
    '''
    docx_data = await docx_data.read()
    docx_data = trans_with_preserving_form(uploaded_file=docx_data, translator=translator, dest_lang=dest_lang)

    # Create an in-memory file object
    file_obj = BytesIO()

    # Save the document to the in-memory file object
    docx_data.save(file_obj)

    # Reset the file object's position to the beginning
    file_obj.seek(0)

    # Return the file as a streamable response
    return StreamingResponse(file_obj, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', headers={'Content-Disposition': 'attachment; filename="document.docx"'})
