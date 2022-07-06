BASE_PATH=/opt/project/translation/mbart-nmt
GPU_ID=2
SRC=ko_KR
TGT=en_XX

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_PATH}/finetune.py --corpus_path ${BASE_PATH}/src/raw_corpus/data_with_upper_lc --plm ${BASE_PATH}/src/plm/reduced_mbart.cc25_v2 --src_lang ${SRC} --tgt_lang ${TGT} --exp_name "reversed-mbart" --max_token_lengh 150 --batch_size 24 --fp16
