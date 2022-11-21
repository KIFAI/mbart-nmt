# fast-MBart(Supports CPU & GPU version)

## Reduction of MBART model size, and boost in inference speed up
  mBART implementation of the Ctranslate2(open source) project (https://github.com/OpenNMT/CTranslate2)
  
  **Pytorch model -> ctranslate2 model -> Quantized ONNX model**
  
---
## Speend And Performance Test with pytorch model

```shell
$ (mbart_env) python test_benchmark.py
```

## Functionalities

- Export any pretrained MBart model to Ctranslate2 easily.
- The exported model supports beam search, batch inference, Multithreading/Parallelism and Asynch API.
- Reduce the model size using quantization.
- X4 Speedup(fp16) compared to PyTorch execution.

## Visualization
<img width="1279" alt="image" src="https://github.com/jyoyogo/mbart-nmt/blob/main/FastModel/Ctrans-MBart/visualize/Mean%20Latency%20by%20seq%20in%20beam(2~5)_beam%23range(2,%206).png?raw=true">
<img width="1279" alt="image" src="https://github.com/jyoyogo/mbart-nmt/blob/main/FastModel/Ctrans-MBart/visualize/Mean%20Score%20by%20seq%20in%20beam(2~5)_beam%23range(2,%206).png?raw=true">
