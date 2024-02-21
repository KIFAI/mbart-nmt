MAIN_PORT=29050
BASE_PATH=/opt/project/mbart-nmt
DEEPSPEED_CONFIG_PATH=/ds_config/zero_stage2_config.json
TOKENIZER_PATH=/src/plm/reduced_hf_mbart06_ko_en_vi_id_km_hi
CORPUS_PATH=/src/train_corpus/cased_corpus_exp
HF_DATASET_ABS_PATH=/src/hf_dataset/dataset_exp
PLM_PATH=/src/plm/reduced_hf_mbart06_ko_en_vi_id_km_hi
PROCESSOR_BATCH_SIZE=20000
NUM_PROC=8
MAX_TOKEN_LENGTH=512
BATCH_SIZE=14
#bf16은 fp16에 비해, 학습속도가 약 10% 빨라지며 gradient OVERFLOW 이슈도 개선
MIXED_PRECISION=bf16
OUTPUT_DIR=output
EXP_NAME=mbart06_ko_en_id_vi_hi_km_large_deepspeed_exp
PACKING_SIZE=512
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-08
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
NUM_WARMUP_STEPS=0
GRADIENT_ACCUMULATION_STEPS=2
NUM_EPOCHS=2
EVAL_CHECK_INTERVAL=0.5
LABEL_SMOOTHING=0.0
EARLY_STOP_METRIC=sacrebleu
PATIENCE=6


#Distributed trainer 사용시 보통 GPU 별로 process를 따로 띄워 사용하기 때문에, Gradient등을 GPU간 공유하기 위해서는 공유 메모리인 /dev/shm에 액세스
#하지만 Docker instance를 띄울 경우 기본값이 64MB로, 아주 작은 값이 할당됨. 
#NCCL의 경우 InfiniBand등 GPU간 전용 통신기기가 발견되지 않을 경우 Default Fallback으로 SHM을 사용하는데, 이로 인해 문제가 발생
#특히, 해당 이슈는 deepspeed 또는 pytorch에서 발생하지 않기 때문에 NCCL Debug를 켜두지 않으면 발견할 수 없음.
#해결책은 로컬환경에서의 동작 또는 도커 컨테이너는 --ipc=host 옵션으로 진행하는 것.

#--ipc=host 옵션은 Docker 컨테이너 내의 프로세스들이 호스트 시스템과 동일한 IPC (Inter-Process Communication) 네임스페이스를 공유하도록 설정하는 옵션.
#IPC는 한 컨테이너 내의 프로세스들이 통신하고 데이터를 공유하는 메커니즘. 이것은 예를 들어 프로세스 간 메모리 공유, 파이프(파이프라인) 통신, 메시지 큐 등을 포함.
#하지만 기본적으로 Docker는 컨테이너 간에 IPC 네임스페이스를 격리.
#--ipc=host 옵션을 사용하면 이러한 IPC 네임스페이스의 격리를 해제하고 호스트 시스템의 IPC 네임스페이스를 사용하게 됨.
#NCCL(NVIDIA Collective Communications Library)은 NVIDIA가 제공하는 GPU 간 통신 라이브러리로, 멀티 GPU 학습 시 GPU 간 데이터 전송에 사용됨.
#NCCL은 IPC 기능을 사용하여 GPU 간에 데이터를 전송하기 때문에, Docker 컨테이너가 IPC 네임스페이스를 호스트와 공유해야 NCCL이 올바르게 작동할 수 있습니다.
#따라서 --ipc=host 옵션을 사용하여 호스트와 동일한 IPC 네임스페이스를 사용하도록 Docker 컨테이너를 설정하면 NCCL 에러를 해결할 수 있음.

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port ${MAIN_PORT} ${BASE_PATH}/finetune_mbart50_no_trainer.py \
    --base_path ${BASE_PATH} \
    --ds_config_path ${BASE_PATH}${DEEPSPEED_CONFIG_PATH} \
    --tokenizer_path ${BASE_PATH}${TOKENIZER_PATH} \
    --corpus_path ${BASE_PATH}${CORPUS_PATH} \
    --hf_dataset_abs_path ${BASE_PATH}${HF_DATASET_ABS_PATH} \
    --plm_path ${BASE_PATH}${PLM_PATH} \
    --processor_batch_size ${PROCESSOR_BATCH_SIZE} \
    --num_proc ${NUM_PROC} \
    --max_token_length ${MAX_TOKEN_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --mixed_precision ${MIXED_PRECISION} \
    --output_dir ${BASE_PATH}/${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
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
    --patience ${PATIENCE}
