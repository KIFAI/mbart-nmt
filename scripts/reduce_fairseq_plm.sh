BASE_PATH=/opt/project/translation/repo/mbart-nmt
SPC_DICT_FN=spiece_changed.vocab
SPC_FN=spiece.model

python ${BASE_PATH}/reduce_fairseq_plm.py --fairseq_plm_path ${BASE_PATH}/src/plm --huggingface_plm 'facebook/mbart-large-cc25' --spc_path ${BASE_PATH}/src/sentencepiece --spc_dict_fn ${SPC_DICT_FN} --spc_fn ${SPC_FN} --reduction_path ${BASE_PATH}/src/plm/reduced_mbart.cc25
