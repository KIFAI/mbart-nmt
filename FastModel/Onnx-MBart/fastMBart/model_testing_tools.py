from time import perf_counter as pc
from matplotlib import pyplot as plt
from transformers import MBart50TokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def evaluate(tokenizer, ref, hyp):
    #Evaluation
    tokenized_ref = tokenizer.tokenize(ref)
    tokenized_hyp = tokenizer.tokenize(hyp)
    score = sentence_bleu([tokenized_ref], tokenized_hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method4)
    return score

def speed_test(
    onnx_model,
    torch_model,
    beam_range: range = range(1, 5, 1),
#    seq_length_range: range = range(10, 500, 50),
    testset=None,
):
    """
        method prints the time took for onnx and pytorch model to finish a text generation task

    args:
        testset (Tuple(str, str)) : text inputs and refs for the model.
        onnx_model : onnx representation of the mbart model,
        torch_model : torch represention of the mbart model,
        beam_range (range) : provide a range, which takes starting end and steps (don't start with 0)
        sequence_length-range (range) : takes the start, end and steps as a range (start with 10)
    return :
        onnx_model_latency : numpy array of latency for each beam number and sequence length
        pytorch_model_latency : numpy array of latency for each beam number and sequence length
    """
    if testset is None:
        srcs = ['i understood it, but will other people get it?', 'they attached a powerpoint file containing the entire education as an appendix.', 'that is, the controller 160 may detect an execution event that corresponds to the loop shortcut item 1110.', 'article 42 of the framework act on national taxes does not prescribe priorities for the property tax obligations of transfer secured creditors.', 'some worry that when the service officially opens early next month, popular stars will flow into the new influencer and dominate the market at once.']
        refs = ['저는 이해했는데 다른 사람들도 그걸 알아챌까요?', '전체 교육 내용을 담은 파워포인트 자료를 부록으로 첨부하였다.', '즉 제어부(160)는 루프 단축 아이템(1110)에 대응하여, 실행 이벤트를 검출할 수 있다.', '국세 기본법 제42조가 양도 담보권자의 물적 납세 의무에 관해 우선순위를 규정하고 있는 것은 아니다.', '일각에서는 다음 달 초 서비스가 정식 오픈되면 인기 스타가 새로운 인플루언서로 유입돼 시장을 단숨에 장악할 것이란 우려의 목소리도 나온다.']
    else :
        if not isinstance(testset, tuple):
            raise TypeError('Check testset. it must be tuple(srcs, refs) format')
        else:
            srcs , refs = testset

    tokenizer = MBart50TokenizerFast.from_pretrained(torch_model.name_or_path)

    xx = []
    yy = []

    for j in beam_range:
        x = []
        y = []
        bleu_x = []
        bleu_y = []

        prev = [1, 2]
        for src, ref in zip(srcs, refs):

            token = tokenizer(
                src,
                return_tensors="pt",
                ).to("cpu")

            input_ids = token["input_ids"]
            attention_mask = token["attention_mask"]

            a = pc()
            o_out = onnx_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                num_beams=j,
            )
            b = pc()
            x.append(b - a)

            c = pc()
            t_out = torch_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                num_beams=j,
            )
            d = pc()
            y.append(d - c)

            print(f"seqL : {len(src)}, onnx-{b-a}, pt-{d-c} .. ONNX faster {(d-c)/(b-a)}")

            onnx_score = evaluate(tokenizer, ref, tokenizer.decode(o_out.squeeze(), skip_special_tokens=True))
            print(f"onnx hyp : {tokenizer.decode(o_out.squeeze(), skip_special_tokens=True)}")
            bleu_x.append(onnx_score)
            torch_score = evaluate(tokenizer, ref, tokenizer.decode(t_out.squeeze(), skip_special_tokens=True))
            print(f"torch hyp : {tokenizer.decode(t_out.squeeze(), skip_special_tokens=True)}")
            bleu_y.append(torch_score)
            print(f"Bleu : onnx-{onnx_score}, pt-{torch_score} .. ONNX higher {onnx_score - torch_score}\n")

            if (t_out.shape[1] == prev[-1]) and (t_out.shape[1] == prev[-2]):
                pass

            prev.append(t_out.shape[1])
        
        mean_x, mean_y = np.mean(x), np.mean(y)
        mean_ratio = mean_y / mean_x
        mean_bleu_x, mean_bleu_y = np.mean(bleu_x), np.mean(bleu_y)
        mean_bleu_ratio = mean_bleu_x / mean_bleu_y
        
        print(f"****Inference Speed at beam no.- {j} onnx : {mean_x} pt : {mean_y} X ratio :{mean_ratio} ****")
        print(f"****Bleu Score at  beam no.- {j} onnx : {mean_bleu_x} pt : {mean_bleu_y} X ratio : {mean_bleu_ratio} ****\n")

        xx.append(x)
        yy.append(y)
        plt.plot(x, "g", y, "r")
        plt.pause(0.05)

    plt.show()
    return np.array(xx), np.array(yy)
