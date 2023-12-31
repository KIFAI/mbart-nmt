BASE_PATH=/opt/project/mbart-nmt
TOKENIZER_PATH=/src/plm/reduced_hf_mbart50_m2m
CORPUS_PATH=/src/train_corpus/cased_corpus_exp
HF_DATASET_PATH=./src/hf_dataset
HF_DATASET_NAME=dataset_exp
NUM_PROC=8
MAX_TOKEN_LENGTH=512
BATCH_SIZE=50000
SRC=en_XX
TGT=ko_KR
PACKING_SIZE=512

python ${BASE_PATH}/prepare_hf_dataset.py \
    --base_path ${BASE_PATH} \
    --tokenizer_path ${BASE_PATH}${TOKENIZER_PATH} \
    --corpus_path ${BASE_PATH}${CORPUS_PATH} \
    --hf_dataset_path ${HF_DATASET_PATH} \
    --hf_dataset_name ${HF_DATASET_NAME} \
    --num_proc ${NUM_PROC} \
    --max_token_length ${MAX_TOKEN_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --drop_case \
    --bi_direction \
    --packing_data \
    --packing_size ${PACKING_SIZE} \
    --hybrid
