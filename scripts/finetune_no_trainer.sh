MAIN_PORT=29050
BASE_PATH=/opt/project/translation/repo/mbart-nmt
DEEPSPEED_CONFIG_PATH=/ds_config/zero_stage2_config.json
TOKENIZER_PATH=/src/plm/reduced_hf_mbart50_m2m_v2
CORPUS_PATH=/src/train_corpus/cased_corpus_exp_v2
HF_DATASET_ABS_PATH=/src/hf_dataset/custom_dataset
PLM_PATH=/src/plm/reduced_hf_mbart50_m2m_v2
PROCESSOR_BATCH_SIZE=20000
NUM_PROC=8
MAX_TOKEN_LENGTH=512
BATCH_SIZE=18
MIXED_PRECISION=fp16
OUTPUT_DIR=output
EXP_NAME=mbart02_enko_exp_large_deepspeed
SRC=en_XX
TGT=ko_KR
PACKING_SIZE=512
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-08
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
NUM_WARMUP_STEPS=0
GRADIENT_ACCUMULATION_STEPS=2
NUM_EPOCHS=2
EVAL_CHECK_INTERVAL=0.25
LABEL_SMOOTHING=0.0
EARLY_STOP_METRIC=sacrebleu
PATIENCE=5

accelerate launch --main_process_port ${MAIN_PORT} ${BASE_PATH}/finetune_mbart50_no_trainer.py \
    --base_path ${BASE_PATH} \
    --ds_config_path ${BASE_PATH}${DEEPSPEED_CONFIG_PATH} \
    --tokenizer_path ${BASE_PATH}${TOKENIZER_PATH} \
    --corpus_path ${BASE_PATH}${CORPUS_PATH} \
    --plm_path ${BASE_PATH}${PLM_PATH} \
    --processor_batch_size ${PROCESSOR_BATCH_SIZE} \
    --num_proc ${NUM_PROC} \
    --max_token_length ${MAX_TOKEN_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --mixed_precision ${MIXED_PRECISION} \
    --output_dir ${BASE_PATH}/${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --drop_case \
    --bi_direction \
    --packing_data \
    --packing_size ${PACKING_SIZE} \
    --hybrid \
    --use_preset \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --adam_epsilon ${ADAM_EPSILON} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_warmup_steps ${NUM_WARMUP_STEPS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_epochs ${NUM_EPOCHS} \
    --eval_check_interval ${EVAL_CHECK_INTERVAL} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --ignore_tokens_ixs_for_loss \
    --early_stop_metric ${EARLY_STOP_METRIC} \
    --patience ${PATIENCE} \
