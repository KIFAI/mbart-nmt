BASE_PATH=/opt/project/translation/repo/mbart-nmt
CORPUS_PATH=src/domain_corpus
GPU_ID=2
SRC=en_XX
TGT=ko_KR
EXP_NAME=domain_mbart

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_PATH}/finetune.py --base_path ${BASE_PATH} --corpus_path ${BASE_PATH}/${CORPUS_PATH} --plm ${BASE_PATH}/src/plm/reduced_mbart.cc25 --src_lang ${SRC} --tgt_lang ${TGT} --exp_name ${EXP_NAME} --max_token_lengh 100 --batch_size 32  --fp16 --additional_special_tokens '법률,경제,교육,문화,예술,관광'
