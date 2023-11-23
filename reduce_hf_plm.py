import re
import argparse
import os, shutil
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sentencepiece import sentencepiece_model_pb2
from transformers import MBartConfig, MBartForConditionalGeneration, MBart50TokenizerFast

LANGS = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

CONTROL_CHRS = ['\n', '\r', '\t']

BT_TAGS = ['<bt>']

SPECIAL_TOKENS_MAP = {
    "additional_special_tokens": LANGS + CONTROL_CHRS + BT_TAGS,
    "bos_token": "<s>",
    "cls_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
    "pad_token": "<pad>",
    "sep_token": "</s>",
    "unk_token": "<unk>"
    }

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default='/opt/project/mbart-nmt',
        type=str,
    )
    parser.add_argument(
        "--plm_name",
        default='facebook/mbart-large-50-many-to-many-mmt',
        type=str,
    )
    parser.add_argument(
        "--plm_local_path",
        default='./src/plm/tmp_ckpt',
        type=str,
    )
    parser.add_argument(
        "--use_pretrained_spm",
        action='store_true',
        help="If you already have pretrained sentenecepiece model, you can use this spm vocab."
    )
    parser.add_argument(
        "--reduction_path",
        default='./src/plm/reduced_hf_mbart50_m2m_v2',
        type=str,
    )
    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
    )

    args = parser.parse_args()

    return args

def prepare_huggingface_plm(plm_name="facebook/mbart-large-50-many-to-many-mmt", save_path="tmp_ckpt"):
    pre_model = MBartForConditionalGeneration.from_pretrained(plm_name)
    pre_model.save_pretrained(save_path)
    pre_tokenizer = MBart50TokenizerFast.from_pretrained(plm_name)
    pre_tokenizer.save_pretrained(save_path)
    pre_config = MBartConfig.from_pretrained(plm_name)

    return pre_config, pre_tokenizer, pre_model

def expand_spm(spm, type_dict={"NORMAL":1, "UNKNOWN":2, "CONTROL":3}, addition_dict={"CONTROL":[("<bt>", 0.0)]}):
    new_piece = type(spm.pieces[0])() #initilaize

    for key, value in addition_dict.items():
        if isinstance(value, list):
            for v_pair in value:
                if not isinstance(v_pair, tuple):
                    raise TypeError("check values should be tuple with piece and score, ex) {'CONTROL':[('<bt>', 0.0)]}")
                else:
                    new_piece.piece = v_pair[0]
                    new_piece.score = v_pair[1]
                    new_piece.type = type_dict[key]
                    spm.pieces.append(new_piece)
                    print(f"Added below\n{new_piece}\n")
        elif isinstance(value, tuple):
            new_piece.piece = value[0]
            new_piece.score = value[1]
            new_piece.type = type_dict[key]
            spm.pieces.append(new_piece)
            print(f"Added below\n{new_piece}\n")
        else:
            raise TypeError("check whether values are UNION[tuple, LIST[tuple]], ex) {'CONTROL':('<bt>', 0.0)}")

    return spm

def filter_vocab(multilingual_vocab, regex_pattern):
    key, value = list(regex_pattern.items())[0]
    tmp_vocab = []
    for i, w in enumerate(multilingual_vocab):
        if value.match(w) is None:
            pass
        else:
            if value.match(w).group() == w:
                tmp_vocab.append(multilingual_vocab[i])
            else:
                pass
    print(f"{key} : {len(tmp_vocab)}")
    return tmp_vocab

def extract_spm_vocab(spm, regex_patterns):
    vocab = ['<unk>','<s>','</s>'] + CONTROL_CHRS + BT_TAGS

    for r_p in regex_patterns:
        vocab += filter_vocab(multilingual_vocab=[p.piece for p in spm.pieces],
                                regex_pattern=r_p)

    filtered_lang_dict = {w.strip() if w.strip() != "" else w:i for i, w in enumerate(set(vocab))}
    print(f" len of vocab : {len(set(vocab))}, len of filtered dict : {len(filtered_lang_dict)}")

    return filtered_lang_dict

def reduce_spm(m, filtered_lang_dict):
    rm_ix2word = {}
    rm_word2id = {}
    for i in tqdm(range(len(m.pieces)), desc="Check necessary token.."):
        word = m.pieces[i].piece
        try:
            _ = filtered_lang_dict[word]
        except Exception as ex:
            rm_ix2word[i] = word
            rm_word2id[word] = i
    print(f"Total num of filtered vocab : {len(m.pieces) - len(rm_ix2word)}")

    for i in tqdm(reversed(list(rm_ix2word.keys())), total=len(rm_ix2word), desc="Remove unnecessary token.."):
        assert m.pieces[i].piece == rm_ix2word[i]
        # print(i, m.pieces[i].piece, rm_ix2word[i])
        m.pieces.pop(i)
    return m

def validate_reduced_spm(spm_path):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2

    sp_old, sp_new = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
    sp_old.load(f"{spm_path}/sentencepiece.bpe.model.old")
    sp_new.load(f"{spm_path}/sentencepiece.bpe.model.new")
    old_result = sp_old.EncodeAsPieces('This eBook is for the use of anyone anywhere at no cost')
    new_result = sp_new.EncodeAsPieces('This eBook is for the use of anyone anywhere at no cost')

    assert old_result==new_result

def load_dict(pre_tokenizer, new_spm):
    pre_dict = {w:i for w, i in sorted(pre_tokenizer.vocab.items(),
                                       key = lambda item: item[1], reverse = False)}
    print(f"Pretrained Tokenizer's vocab size : {len(pre_tokenizer.get_vocab())}")
    new_dict = {}

    reduced_pieces = [p.piece for p in new_spm.pieces]

    initialized_vocab_num = 0
    common_vocab_num = 0

    for token in list(set(reduced_pieces + [SPECIAL_TOKENS_MAP['mask_token']] + [SPECIAL_TOKENS_MAP['pad_token']] + LANGS + CONTROL_CHRS + BT_TAGS)):
        try:
            new_dict[token] = pre_dict[token]
            common_vocab_num += 1
        except Exception as ex:
            #if token not in pre_dict, it's id is changed as <unk> id. it will be initialized in embed token, unk
            new_dict[token] = pre_dict['<unk>']
            initialized_vocab_num += 1

    print(f"Common word's len between reduced dict and pre dict : {common_vocab_num}")
    new_dict = {w:i for w, i in sorted(new_dict.items(),
                                       key = lambda item: item[1], reverse = False)}
    if initialized_vocab_num > 0 :
        print(f"Newly added word's number in vocab : {initialized_vocab_num}")

    return pre_dict, new_dict

def reduce_plm(pre_config, pre_model, pre_dict, new_dict,
               plm_local_path = "tmp_ckpt", reduced_model_path='reduced_mbart50'):

    org_sd = pre_model.state_dict()
    resized_sd = pre_model.state_dict()

    print("Reducing Mbart PLM.....\n")
    print(f"Reduced vocab size : {len(new_dict)}")
    print(Counter(list(new_dict)).most_common()[:5])
    for name in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "model.shared.weight", "lm_head.weight"]:
        pre_tensor: torch.Tensor = org_sd[name]
        new_tensor = torch.zeros(
            [len(new_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for new_i, pre_i in enumerate(new_dict.values()):
            new_tensor[new_i] = pre_tensor[pre_i]
        resized_sd[name] = new_tensor
    resized_sd["final_logits_bias"] = resized_sd["final_logits_bias"][:, :len(new_dict)]

    print("****** Check pre mbart vocab size ******\n")
    print(pre_config)

    pre_config.vocab_size = len(new_dict)
    new_model = MBartForConditionalGeneration.from_pretrained(None, config=pre_config, state_dict=resized_sd)

    if os.path.exists(reduced_model_path):
        print(f'reduction path, {reduced_model_path} already exsits\n')
    else:
        print('make reduction path..\n')
        os.mkdir(reduced_model_path)

    new_model.config._name_or_path = args.reduction_path
    new_model.config.max_length = args.max_length

    new_model.save_pretrained(reduced_model_path)
    new_config = MBartConfig.from_pretrained(reduced_model_path)
    print(new_config)


    shutil.copy(f'{plm_local_path}/sentencepiece.bpe.model.new',
                f'{plm_local_path}/sentencepiece.bpe.model')
    os.remove(f'{plm_local_path}/tokenizer.json')

    new_tokenizer = MBart50TokenizerFast.from_pretrained(plm_local_path)

    print(f"New model's vocab size : {new_config.vocab_size}, New tokenizer's vocab size : {new_tokenizer.vocab_size}")
    assert new_config.vocab_size == new_tokenizer.vocab_size

    new_tokenizer.add_special_tokens({"additional_special_tokens" : LANGS + CONTROL_CHRS + BT_TAGS})
    new_tokenizer.save_pretrained(reduced_model_path)

    import json
    with open(os.path.join(reduced_model_path, 'tokenizer_config.json'), 'r') as f:
        tokenizer_config = json.load(f)
    tokenizer_config['additional_special_tokens'] = new_tokenizer.additional_special_tokens
    tokenizer_config['name_or_path'] = reduced_model_path
    tokenizer_config['special_tokens_map_file'] = os.path.join(reduced_model_path, 'special_tokens_map.json')

    with open(os.path.join(reduced_model_path, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"New tokenizer's vocab size : {new_tokenizer.vocab_size}")
    assert len(new_dict) == new_tokenizer.vocab_size

    shutil.rmtree(plm_local_path)

def test_uni_trans(reduction_path):
    new_model = MBartForConditionalGeneration.from_pretrained(reduction_path)
    new_tokenizer = MBart50TokenizerFast.from_pretrained(reduction_path)
    #Test uni directional translation
    test_set = [['I understood it, but will other people get it?', 'en_XX', 'ko_KR'],
               ['저는 이해했는데 다른 사람들도 그걸 알아챌까요?', 'ko_KR', 'en_XX'],
               ['I understood it, but will other people get it?', 'en_XX', 'zh_CN'],
               ['我明白了，但其他人会明白吗？', 'zh_CN', 'en_XX'],
               ['저는 이해했는데 다른 사람들도 그걸 알아챌까요?', 'ko_KR', 'zh_CN'],
               ['我明白了，但其他人会明白吗？', 'zh_CN', 'ko_KR'],]

    for src, src_lang, tgt_lang in test_set:
        print(f"Source sentence : {src}")
        new_tokenizer.src_lang = src_lang
        encoded = new_tokenizer(src, return_tensors="pt")
        generated_tokens = new_model.generate(**encoded,
                                          forced_bos_token_id=new_tokenizer.lang_code_to_id[tgt_lang]
        )
        print(f'Tgt lang : {tgt_lang} / {new_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)}')
        print('\n')

    sample = '2020년 6월 한 경찰관이 제로지 플로이드를 살해한 직후, 제프 베조스는 BLM을 옹호했다'
    assert new_tokenizer.decode(new_tokenizer(sample).input_ids, skip_special_tokens=True) == ''.join(new_tokenizer.tokenize(sample)).replace('▁', ' ').strip()

def main(args):
    pre_config, pre_tokenizer, pre_model = prepare_huggingface_plm(plm_name=args.plm_name,
                                                       save_path=args.plm_local_path)
    pre_spm = sentencepiece_model_pb2.ModelProto()

    if args.use_pretrained_spm:
        pre_spm.ParseFromString(open(os.path.join(args.base_path, "src/sentencepiece/spiece.model"), 'rb').read())
        regex_patterns = [{'all':re.compile(r'.*')}]
    else:
        '''
        If you extract specific lang's words in pretrained mBart50 vocab,
        you should define regex pattern you want select.
        And if '▁' is omitted in vocab, it will be generated '<unk>' token
        in case of encoding sequence, like "['▁', '2', '0', '2', '0', '년', ~~]"
        '''
        pre_spm.ParseFromString(open(f"{args.plm_local_path}/sentencepiece.bpe.model", 'rb').read())
        regex_patterns = [
            {'num':re.compile(r'[0-9]')},
            {'punc':re.compile(r"^▁?[!\"#$%&\\'\(\)*\+,\-\./:;<=>\?@\[\]\^_▁`{\|}~]$")},
            {'ko':re.compile("▁?[\uAC00-\uD7AF|\u1100-\u11FF|\uA960-\uA97F|\uD7B0-\uD7FF|\u3130-\u318F]+")},
            {'en':re.compile(r'▁?[a-zA-Z]+')},
            #{'hanja':re.compile("▁?[\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff]+")}
        ]

    pre_spm = expand_spm(spm=pre_spm, type_dict={"NORMAL":1, "UNKNOWN":2, "CONTROL":3}, 
                        addition_dict={"CONTROL": [(el,0.0) for el in CONTROL_CHRS+BT_TAGS]})
    filtered_lang_dict = extract_spm_vocab(spm=pre_spm, regex_patterns=regex_patterns)
    new_spm = reduce_spm(pre_spm, filtered_lang_dict)

    # Backup the old model
    Path(f"{args.plm_local_path}/sentencepiece.bpe.model").rename(f"{args.plm_local_path}/sentencepiece.bpe.model.old")
    # Write the new model to disk
    with open(f"{args.plm_local_path}/sentencepiece.bpe.model.new", 'wb') as f:
        f.write(new_spm.SerializeToString())
    # Validate
    validate_reduced_spm(args.plm_local_path)

    pre_dict, new_dict = load_dict(pre_tokenizer, new_spm)
    reduce_plm(pre_config, pre_model, pre_dict, new_dict,
          plm_local_path = args.plm_local_path,
          reduced_model_path=args.reduction_path)
    test_uni_trans(reduction_path = args.reduction_path)

if __name__ == '__main__':
    args = define_argparser()
    main(args)
