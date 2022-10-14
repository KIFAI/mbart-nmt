import ctranslate2
import os, time
import numpy as np
from time import perf_counter as pc
from matplotlib import pyplot as plt
from transformers import MBartTokenizer, MBart50TokenizerFast, MBartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def validate_vocab(ctrans_path):
    with open(f"{ctrans_path}/shared_vocabulary.txt", "r") as f:
        vocab = f.readlines()

    vocab = [s.strip() for s in vocab]

    base_token = ['<s>', '<pad>', '</s>', '<unk>']

    if sum([True for t in base_token if t in vocab]) ==4 :
        print(f"{base_token} tokens are checked in shared vocab")
    else:
        ommition_case = [t for t in base_token if t not in vocab]
        raise ValueError("Check {ommition_case} tokens in vocab")

    base_token_ixs = [i for i, w in enumerate(vocab) if w in base_token]
    if base_token_ixs != [0,1,2,3]:
        print("Change vocab's order normally...")
        for b_t in reversed(base_token):
            print(f"popped (token : {b_t}, ix : {vocab.index(b_t)})")
            vocab.pop(vocab.index(b_t))
        for b_t in reversed(base_token):
            vocab.insert(0, b_t)
        print(f"Vocab's order is changed normally, as like {vocab[:4]}")
        assert vocab[:4] == base_token
    else:
        print("Vocab's order is checked, normally")

    with open(f"{ctrans_path}/shared_vocabulary.txt", "w") as f:
        f.write("\n".join(vocab))

def evaluate(tokenizer, ref, hyp):
    #Evaluation
    tokenized_ref = tokenizer.tokenize(ref)
    tokenized_hyp = tokenizer.tokenize(hyp)
    score = sentence_bleu([tokenized_ref], tokenized_hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method4)
    return score

def plotting(exp_name, beam_size, seq_len_list, 
             ctrans_results, torch_results, device):
    #### 1. bar plot으로 나타낼 데이터 입력
    models = ['Ctrans', 'Pytorch']
    xticks = seq_len_list
    data = {'Ctrans':ctrans_results,
            'Pytorch':torch_results}

    #### 2. matplotlib의 figure 및 axis 설정
    fig, ax = plt.subplots(1,1,figsize=(10,5)) # 1x1 figure matrix 생성, 가로(7인치)x세로(5인치) 크기지정
    colors = ['orange', 'salmon']
    width = 0.15
    
    #### 3. bar 그리기
    for i, model in enumerate(models):
        pos = compute_pos(xticks, width, i, models)
        bar = ax.bar(pos, data[model], width=width*0.95, label=model, color=colors[i])
        present_height(ax, bar) # bar높이 출력

    #### 4. x축 세부설정
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=10)	
    ax.set_xlabel(f'Source sequence length in beam#{beam_size} at {device}', fontsize=14)

    #### 5. y축 세부설정
    yticks = round(max(ctrans_results + torch_results) / 5, 2)
    #ax.set_ylim([0.5,0.76])
    ax.set_yticks([yticks*c for c in range(0,7,1)])
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_ylabel(f'{exp_name}', fontsize=14)

    #### 6. 범례 나타내기
    ax.legend(loc='upper left', shadow=True, ncol=1)

    #### 7. 보조선(눈금선) 나타내기
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

    #### 8. 그래프 저장하고 출력하기
    plt.tight_layout()
    if os.path.exists('visualize'):
        pass
    else:
        os.mkdir('visualize')
    plt.savefig(f'./visualize/{exp_name}_beam#{beam_size}.png', format='png', dpi=300)
    plt.show()

def compute_pos(xticks, width, i, models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i-0.5*(n-1)
    return index + width*correction

def present_height(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x()+rect.get_width()*0.5
        posy = height*1.01
        ax.text(posx, posy, '%.3f' % height, rotation=90, ha='center', va='bottom')
        
def speed_test(
    ctrans_model,
    torch_model,
    tokenizer,
    beam_range: range = range(2, 6, 1),
    warmup_range: range = range(0, 10, 1),
    inference_range: range = range(0, 10, 1),
    testset=None,
    device="cpu"
):
    """
        method prints the time took for ctrans and pytorch model to finish a text generation task

    args:
        testset (Tuple(str, str)) : text inputs and refs for the model.
        ctrans_ver1 : ctrans representation of the mbart model(ideal condition with qunatization),
        ctrans_ver2 : ctrans representation of the mbart model(same condition with torch),
        torch_model : torch represention of the mbart model,
        beam_range (range) : provide a range, which takes starting end and steps (don't start with 0)
        sequence_length-range (range) : takes the start, end and steps as a range (start with 10)
    return :
        ctrans_ver1_latency : numpy array of latency for each beam number and sequence length
        ctrans_ver2_latency : numpy array of latency for each beam number and sequence length
        pytorch_model_latency : numpy array of latency for each beam number and sequence length
        ctrans_ver1_bleu : numpy array of latency for each beam number and sequence length
        ctrans_ver2_bleu : numpy array of latency for each beam number and sequence length
        pytorch_model_bleu : numpy array of latency for each beam number and sequence length
    """
    if testset is None:
        srcs = ['i understood it, but will other people get it?'
                ,'that is, the controller 160 may detect an execution event that corresponds to the loop shortcut item 1110.'
                ,'the fact that easements and mortgages, which are property rights which do not have the right to occupy, usually do not have the right to claim the return of the home, it is interesting from the historic point of view of the property claim.'
                ,'in table ii-12, the price of parcels traded in november 2009, which was 13,608 won per square meter, rises to 48,911 won per square meter on december 7, 2009, less than a month later.'
                ,'the korea drone football association (chairman kim seung-soo, jeonju mayor) signed an mou on the 14th to revitalize the domestic competition of drone soccer developed and distributed by the korea model aviation association (chairman park chan-deok) and jeonju city for the first time in the world and expand the driving force for demonstration games at international competitions.'
                ,'an automatic fishing system includes a plurality of fishing machines each of which has rotation drums capable of winding fishing lines and capable of rotating in a reeling direction and an unreeling direction, a drive motor for driving the rotation drums in the reeling direction and the unreeling direction, and an electromagnetic clutch for transmitting rotation of the drive motor to the rotation drums, a hooking detection means for always detecting hooking of fish at any one of the plurality of fishing machines, and a fishing machine control means for controlling operations of the plurality of fishing machines in accordance with detection result of the hooking detection means. the fishing machine control means controls the rotation drums of other fishing machines that hook no fish in the reeling direction so that hooks of the other fishing machines are wound up to a water surface or near the water surface.']
        refs = ['저는 이해했는데 다른 사람들도 그걸 알아챌까요?'
                ,'즉 제어부(160)는 루프 단축 아이템(1110)에 대응하여, 실행 이벤트를 검출할 수 있다.'
                ,'점유권능을 가지지 않는 물권인 지역권과 저당권이 물권적 반환청구권을 가지지 않는다는 점은 물권적 청구 권의 역사적 관점에서 볼 때에는 흥미로운 점이라고 할 수 있다.'
                ,'<표 Ⅱ-12>에서 보는 바와 같이, 2009년 11월에 거래된 필지의 제곱미터당 13,608원 이었던 가격은 한달도 지나지 않은 2009년 12월 7일에 제곱미터당 48,911원으로 상승한다.'
                ,'(사)대한드론축구협회(협회장 김승수 전주시장)은 14일 한국모형항공협회(회장 박찬덕)와 전주시가 세계 최초로 개발하고 보급한 드론축구의 국내대회 활성화와 국제대회 시범경기 추진동력을 확대하기 위한 업무협약(MOU)을 체결했다.'
                ,'선상에 탑재된 복수의 낚시 기기를 집중 제어하고, 물고기가 걸려있는 경우, 서로의 낚싯줄 및 낚싯바늘의 엉킴을 신속하게 방지하는 것이 가능한 자동 물고기 낚시 시스템 및 자동 물고기 낚시 방법을 제공한다. 자동 물고기 낚시 시스템은, 낚싯줄을 감아 놓아, 권하 방향 및 권상 방향으로 회전 가능한 회전 드럼과, 회전 드럼을 권하 방향 또는 권상 방향으로 구동시키는 구동 모터와, 구동 모터의 회전을 회전 드럼에 전달하는 전자 클러치를 각각 구비하고 있는 복수의 낚시 기기와, 복수의 낚시 기기 중 어느 하나에 물고기가 걸린 것을 항상 검출하고 있는 어획 검출 수단과, 어획 검출 수단의 검출 결과에 기초하여, 복수의 낚시 기기의 동작을 제어하는 낚시 기기 제어 수단을 구비하고 있다. 낚시 기기 제어 수단은, 어획 검출 수단에 의해 복수의 낚시 기기 중 어느 하나에 물고기가 걸린 것이 검출되는 경우는 즉시 물고기가 걸리지 않은 낚시 기기의 회전 드럼을 권상 방향으로 회전시켜 낚싯바늘을 해수면 또는 해수면 근방까지 감아 올리도록 제어한다.']
    else :
        if not isinstance(testset, tuple):
            raise TypeError('Check testset. it must be tuple(srcs, refs) format')
        else:
            srcs , refs = testset

    latency_xx, latency_yy = [], []
    bleu_xx, bleu_yy = [], []

    for j in beam_range:
        latency_x, latency_y = [], []
        bleu_x, bleu_y = [], []
        src_seq_len = []

        for src, ref in zip(srcs, refs):

            token = tokenizer(src, return_tensors="pt").to(device)
            input_ids, attention_mask = token["input_ids"], token["attention_mask"]
            
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(src))
            src_seq_len.append(len(source))
            
            warmup = [ctrans_model.translate_batch(source=[source],target_prefix=[['ko_KR']],beam_size=j) for i in warmup_range]
            a = pc()
            out = [ctrans_model.translate_batch(source=[source],target_prefix=[['ko_KR']],beam_size=j)
                   for i in inference_range][0]
            b = pc()
            ctrans_latency = (b - a)/len(inference_range)
            latency_x.append(ctrans_latency)
            
            if device != 'cpu':
                torch_model.eval()
                torch_model.half()
            
            warmup = [torch_model.generate(input_ids=input_ids,attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],num_beams=j)
                 for i in warmup_range]
            c = pc()
            o = [torch_model.generate(input_ids=input_ids,attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],num_beams=j)
                 for i in inference_range][0]
            d = pc()
            pytorch_latency = (d - c)/len(inference_range)
            latency_y.append(pytorch_latency)

            print(f"seqL : {len(source)}, Ctrans-{ctrans_latency}, pt_fp16-{pytorch_latency}")
            print(f"Ctrans faster {pytorch_latency/ctrans_latency} than Pytorch_fp16")
            
            ctrans_score = evaluate(tokenizer, ref, tokenizer.decode(tokenizer.convert_tokens_to_ids(out[0].hypotheses[0]), skip_special_tokens=True))
            bleu_x.append(ctrans_score)
            torch_score = evaluate(tokenizer, ref, tokenizer.decode(o.squeeze(), skip_special_tokens=True))
            bleu_y.append(torch_score)
            print(f"ref out : {ref}")
            print(f"ctran out : {tokenizer.decode(tokenizer.convert_tokens_to_ids(out[0].hypotheses[0]), skip_special_tokens=True)}")
            print(f"torch out : {tokenizer.decode(o.squeeze(), skip_special_tokens=True)}")
            #print(f"Bleu : Ctrans_int16-{bleu_score_v1}, Ctrans_fp16-{bleu_score_v2}, pt-{torch_score}")
            #print(f"Ctrans_fp16 higher {bleu_score_v2 - torch_score} than Pytorch_fp16")
            #print(f"Ctrans_int16 higher {bleu_score_v1 - torch_score} than Pytorch_fp16\n")

        mean_x, mean_y = np.mean(latency_x), np.mean(latency_y)
        mean_ratio = mean_y / mean_x
        mean_bleu_x, mean_bleu_y = np.mean(bleu_x), np.mean(bleu_y)
        mean_bleu_ratio = mean_bleu_x / mean_bleu_y
        
        print(f"****Inference Speed at beam no.- {j} Ctrans : {mean_x} pt : {mean_y}****")
        print(f"Ctrans ratio :{mean_ratio}")
        print(f"****Bleu Score at beam no.- {j} Ctrans : {mean_bleu_x} pt : {mean_bleu_y}****")
        print(f"Ctrans ratio :{mean_bleu_ratio}\n")

        plotting('Inference Latency', beam_size=j,seq_len_list=src_seq_len, 
                 ctrans_results=latency_x, torch_results=latency_y, device=device)
        plotting('Bleu Score(N-gram)', beam_size=j,seq_len_list=src_seq_len, 
                 ctrans_results=bleu_x, torch_results=bleu_y, device=device)
        
        latency_xx.append(latency_x)
        latency_yy.append(latency_y)
        bleu_xx.append(bleu_x)
        bleu_yy.append(bleu_y)

    mean_latency_x = np.mean(np.reshape(latency_xx, (len(beam_range),len(srcs))), axis=0)
    mean_latency_y = np.mean(np.reshape(latency_yy, (len(beam_range),len(srcs))), axis=0)
    mean_bleu_x = np.mean(np.reshape(bleu_xx, (len(beam_range),len(srcs))), axis=0)
    mean_bleu_y = np.mean(np.reshape(bleu_yy, (len(beam_range),len(srcs))), axis=0)

    plotting('Mean Latency(int16) by seq in beam(2~5)', beam_size=beam_range,seq_len_list=src_seq_len, 
             ctrans_results=mean_latency_x, torch_results=mean_latency_y, device=device)
    print(f"Ctrans faster by seq : {np.array(mean_latency_y) / np.array(mean_latency_x)}\n")
          
    plotting('Mean Score by seq in beam(2~5)', beam_size=beam_range,seq_len_list=src_seq_len, 
             ctrans_results=mean_bleu_x, torch_results=mean_bleu_y, device=device)
    print(f"Ctrans bleu higher by seq : {np.array(mean_bleu_x) / np.array(mean_bleu_y)}\n")
    

    return np.array(latency_xx), np.array(latency_yy), np.array(bleu_xx), np.array(bleu_yy)

if __name__ == "__main__":
    device = 'cuda'
    ctrans_index, torch_index = 4,5 
    #plm_path = '/opt/project/translation/repo/mbart-nmt/src/ftm/reduced_hf_mbart50_m2m'
    plm_path = '/opt/project/translation/repo/mbart-nmt/src/ftm/cased_mbart50-finetuned-en_XX-to-ko_KR/checkpoint-61500'
    ctrans_path = './ctrans_fp16'
    
    converter = ctranslate2.converters.TransformersConverter(plm_path)
    converter.convert(ctrans_path, force=True, quantization='float16')
    validate_vocab(ctrans_path)

    ctrans_model = ctranslate2.Translator(ctrans_path, inter_threads=1, 
            device=f"{device}" if device == "cuda" else "cpu",
            device_index=[0] if device=="cpu" else [ctrans_index])
    pytorch_model = MBartForConditionalGeneration.from_pretrained(plm_path, use_cache=True).to(f"{device}:{torch_index}")

    tokenizer = MBart50TokenizerFast.from_pretrained(pytorch_model.name_or_path)
    tokenizer.src_lang = "en_XX"
    
    speed_test(ctrans_model=ctrans_model, torch_model=pytorch_model, tokenizer=tokenizer,
            warmup_range = range(0, 10, 1), inference_range = range(0, 10, 1), device=f"{device}:{torch_index}")
