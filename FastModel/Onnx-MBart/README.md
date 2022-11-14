# fast-MBart

### Reduction of MBART model size, and boost in inference speed up
  MBART implementation of the fastT5 library (https://github.com/Ki6an/fastT5)
  
  **Pytorch model -> ONNX model -> Quantized ONNX model**

---
## Install

Install using requirements.txt file
```shell
$ python3 -m venv onnx_env
$ . onnx_env/bin/activate
(onnx_env)$ pip install --upgrade pip
(onnx_env)$ pip install -r requirements.txt

---
## Usage

The `export_and_get_onnx_model()` method exports the given pretrained MBart model to onnx, quantizes it and runs it on the onnxruntime with default settings. The returned model from this method supports the `generate()` method of huggingface.

> If you don't wish to quantize the model then use `quantized=False` in the method.

```python
from fastMBart import export_and_get_onnx_model
from transformers import MBartTokenizer

model_name = 'facebook/mbart-large-en-ro'
model = export_and_get_onnx_model(model_name)

tokenizer = MBartTokenizer.from_pretrained(model_name)
input = "This is a english sentence and needs to be translated."
token = tokenizer(input, return_tensors='pt')

tokens = model.generate(input_ids=token['input_ids'],
               attention_mask=token['attention_mask'],
               num_beams=3,
	       decoder_start_token_id=tokenizer.lang_code_to_id)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
print(output)
```

> to run the already exported model use `get_onnx_model()`

you can customize the whole pipeline as shown in the below code example:

```python
from fastMBart import (OnnxMBart, get_onnx_runtime_sessions,
                    generate_onnx_representation, quantize)
from transformers import MBartTokenizer

model_or_model_path = 'facebook/mbart-large-en-ro'

# Step 1. convert huggingfaces bart model to onnx
onnx_model_paths = generate_onnx_representation(model_or_model_path)

# Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
# The process is slow for the decoder and init-decoder onnx files (can take up to 15 mins)
quant_model_paths = quantize(onnx_model_paths)

# step 3. setup onnx runtime
model_sessions = get_onnx_runtime_sessions(quant_model_paths)

# step 4. get the onnx model
model = OnnxMBart(model_or_model_path, model_sessions)

                      ...
```
##### custom output paths 
By default, fastMBart creates a `models-mbart` folder in the current directory and stores all the models. You can provide a custom path for a folder to store the exported models. And to run already `exported models` that are stored in a custom folder path: use `get_onnx_model(onnx_models_path="/path/to/custom/folder/")`

```python
from fastMBart import export_and_get_onnx_model, get_onnx_model

model_name = "facebook/mbart-large-en-ro"
custom_output_path = "/path/to/custom/folder/"

# 1. stores models to custom_output_path
model = export_and_get_onnx_model(model_name, custom_output_path)

# 2. run already exported models that are stored in custom path
# model = get_onnx_model(model_name, custom_output_path)
```
## Functionalities

- Export any pretrained MBart model to ONNX easily.
- The exported model supports beam search and greedy search and more via `generate()` method.
- Reduce the model size using quantization.
- Speedup compared to PyTorch execution for greedy search and for beam search.

