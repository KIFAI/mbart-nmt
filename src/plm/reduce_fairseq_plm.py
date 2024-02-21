import os
import argparse
import wget
import tarfile
import shutil
import glob
import torch
from collections import Counter
from fairseq.data import Dictionary
from transformers import (
    MBartForConditionalGeneration, MBartTokenizer, MBartConfig
)
from typing import List
def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fairseq_plm_path",
        default='./src/plm',
        type=str,
    )
    parser.add_argument(
        "--huggingface_plm",
        default='facebook/mbart-large-cc25',
        type=str,
    )
    parser.add_argument(
        "--spc_path",
        default='./src/sentencepiece',
        type=str,
    )
    parser.add_argument(
        "--spc_dict_fn",
        default='custom_spm_mbart_vocab.txt',
        type=str,
    )
    parser.add_argument(
        "--spc_fn",
        default='custom_spm.model',
        type=str,
    )
    parser.add_argument(
        "--reduction_path",
        default='./src/plm/reduced_mbart.cc25',
        type=str,
    )

    args = parser.parse_args()

    return args

def prepare_fairseq_plm(args):
    '''
    Download the pre-trained model directly from fairseq instead of huggingface becuase you will need weight reduction tasks.
    '''
    if os.path.isfile(f'{args.fairseq_plm_path}/mbart.cc25.v2.tar.gz'):
        pass
    else:
        print(f"Download mbart.cc25 PLM")
        try:
            os.mkdir(f'{args.fairseq_plm_path}')
        except Exception as ex:
            print(ex)
        url = "https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz"
        wget.download(url, out=args.fairseq_plm_path)

    if os.path.isdir(f'{args.fairseq_plm_path}/mbart.cc25.v2'):
        print('Mbart.cc25 PLM already exists\n')
    else:
        print('Extracting mbart plm tar file...\n')
        tar_file = tarfile.open(f'{args.fairseq_plm_path}/mbart.cc25.v2.tar.gz')
        tar_file.extractall(path=args.fairseq_plm_path)
        tar_file.close()

def load_dict(langs: List[str], path: str) -> Dictionary:
        print(f'Load {path}')
        d = Dictionary.load(path)
        for ll in langs:
            d.add_symbol(f"[{ll}]")
        d.add_symbol("<mask>")
        d.add_symbol("<pad>")
        return d

def reduce_plm(pre_dict, ft_dict):
    '''
    This weight reduction process is performed to solve the problem that the size of the base model is huge
    '''
    
    print("Preparing huggingface's format plm files.....\n")
    model = MBartForConditionalGeneration.from_pretrained(f"{args.huggingface_plm}")
    org_sd = model.state_dict()
    resized_sd = model.state_dict()

    print("Reducing Mbart PLM.....\n")
    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))
    print(Counter(mapping).most_common()[:5])
    for name in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "model.shared.weight", "lm_head.weight"]:
        pre_tensor: torch.Tensor = org_sd[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        resized_sd[name] = ft_tensor
    resized_sd["final_logits_bias"] = resized_sd["final_logits_bias"][:, :len(ft_dict)]

    pre_config = MBartConfig.from_pretrained(f"{args.huggingface_plm}")
    print("****** Check pre mbart vocab size ******\n")
    print(pre_config)
    pre_config.vocab_size = len(ft_dict)
    new_model = MBartForConditionalGeneration.from_pretrained(None, config=pre_config, state_dict=resized_sd)

    if os.path.exists(args.reduction_path):
        print('reduction path already exsits\n')
    else:
        print('make reduction path..\n')
        os.mkdir(args.reduction_path)

    new_model.save_pretrained(args.reduction_path)
    ft_config = MBartConfig.from_pretrained(args.reduction_path)
    print("****** Check ft custom mbart vocab size ******\n")
    print(ft_config)

def prepare_tokenizer():
    '''
    Overwrite subword tokenizer file with the one you created earlier
    '''
    print("Preparing huggingface's format tokenizer files.....\n")
    tokenizer = MBartTokenizer.from_pretrained(f"{args.huggingface_plm}")
    tokenizer.save_pretrained(args.reduction_path)
        
    shutil.copy(os.path.join(args.spc_path, args.spc_fn), os.path.join(args.reduction_path, 'sentencepiece.bpe.model'))
    print("Now both the model and the tokenizer can be called from the reduced_model directory.")
    print(glob.glob(f"{args.reduction_path}/*"))

def main(args):
    prepare_fairseq_plm(args)
    langs = ["ar_AR","cs_CZ","de_DE","en_XX","es_XX","et_EE","fi_FI","fr_XX","gu_IN","hi_IN","it_IT","ja_XX",
            "kk_KZ","ko_KR","lt_LT","lv_LV","my_MM","ne_NP","nl_XX","ro_RO","ru_RU","si_LK","tr_TR","vi_VN","zh_CN"]
    pre_dict = load_dict(langs, f"{args.fairseq_plm_path}/mbart.cc25.v2/dict.txt")
    ft_dict = load_dict(langs, os.path.join(args.spc_path, args.spc_dict_fn))
    reduce_plm(pre_dict, ft_dict)
    prepare_tokenizer()
    

if __name__ == '__main__':
    args = define_argparser()
    main(args)
