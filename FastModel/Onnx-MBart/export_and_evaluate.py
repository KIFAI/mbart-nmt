import os, argparse
from pathlib import Path
from fastMBart import (OnnxMBart, get_onnx_runtime_sessions,
                    generate_onnx_representation, quantize, speed_test)
from fastMBart.onnx_exporter import get_model_paths
from transformers import MBartForConditionalGeneration

def define_argparser():
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath('./')))
    model_path = os.path.join(repo_path, 'src/ftm/cased_mbart50-finetuned-en_XX-to-ko_KR/final_checkpoint')

    p = argparse.ArgumentParser()
    p.add_argument('--plm_path', type=str, default=f'{model_path}')
    p.add_argument('--onnx_path', type=str, default='models-mbart')
    p.add_argument('--export', default=False, action='store_true')
    
    config = p.parse_args()

    return config

if __name__ == '__main__':
    config = define_argparser()
    
    if config.export :
        # Step 1. convert huggingfaces bart model to onnx
        onnx_model_paths = generate_onnx_representation(config.plm_path)

        # Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
        # The process is slow for the decoder and init-decoder onnx files (can take up to 15 mins)
        quant_model_paths = quantize(onnx_model_paths)
    else:
        quant_model_paths = get_model_paths(config.plm_path, Path(config.onnx_path), quantized=True)

    # step 3. setup onnx runtime
    model_sessions = get_onnx_runtime_sessions(model_paths=quant_model_paths, provider=["CPUExecutionProvider"])

    # step 4. get the onnx model
    quantized_onnx_model = OnnxMBart(config.plm_path, model_sessions)

    # step 5. evaluate
    pytorch_model = MBartForConditionalGeneration.from_pretrained(config.plm_path, use_cache=True)
    pytorch_model.eval()
    speed_test(onnx_model = quantized_onnx_model,
            torch_model = pytorch_model,
            beam_range = range(1, 5, 1))
