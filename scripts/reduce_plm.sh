BASE_PATH=/opt/project/translation/repo/mbart-nmt

python ${BASE_PATH}/reduce_fairseq_plm.py --fairseq_plm_path ${BASE_PATH}/src/plm --huggingface_plm 'facebook/mbart-large-cc25' --spc_path ${BASE_PATH}/src/sentencepiece --spc_dict_fn 'mono.lower_aihub_mbart.vocab' --spc_fn 'mono.lower_aihub.model' --reduction_path ${BASE_PATH}/src/plm/reduced_mbart.cc25
