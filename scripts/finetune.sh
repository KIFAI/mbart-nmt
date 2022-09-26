BASE_PATH=/opt/project/translation/repo/mbart-nmt
CORPUS_PATH=src/train_corpus/cased_corpus_v2
GPU_ID='1,3,4,0'
SRC=en_XX
TGT=ko_KR
EXP_NAME=cased_mbart50_v3

CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_PATH}/finetune_mbart50.py --base_path ${BASE_PATH} --corpus_path ${BASE_PATH}/${CORPUS_PATH} --plm ${BASE_PATH}/src/plm/reduced_hf_mbart50_m2m_v2 --src_lang ${SRC} --tgt_lang ${TGT} --exp_name ${EXP_NAME} --max_token_length 256 --drop_case --batch_size 8 --gradient_accumulation_steps 2 --num_epochs 2 --fp16 --bi_direction --packing_data --hybrid
