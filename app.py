import time
from flask import Flask, request
from flask_cors import CORS
from transformers import (MBartForConditionalGeneration, MBartTokenizer)


model_path = "./src/ftm/mbart-custom-spm-finetuned-en_XX-to-ko_KR/final_checkpoint"
tokenizer = MBartTokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path).to(f"cuda:0")
model.half()

app = Flask("api_test")
CORS(app)
@app.route('/', methods=['POST'])
def hello():
    sources = request.get_json()
    print(sources)
    
    start = time.time()
    inputs = tokenizer([sources['q'].upper()], return_tensors="pt", padding=True).to(f'cuda:0')
    translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], num_beams=3, early_stopping=True, max_length=150)
    pred = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    end = time.time()
    print(f"pred: {pred}\nElaped time : {end-start}")
    return {'translatedText' : pred[0]}

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
