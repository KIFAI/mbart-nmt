BASE_PATH=/opt/project/translation/mbart-nmt
EXP_NAME=mbart-custom-spm
SAMPLE_N=3111
BATCH_N=64
GPU_ID=2

python ${BASE_PATH}/evaluate.py --exp_name ${EXP_NAME} --sample_n ${SAMPLE_N} --batch_size ${BATCH_N} --cuda_id ${GPU_ID} --fp16 > /opt/project/translation/mbart-nmt/src/hypothesis/${EXP_NAME}_lc_hypothesis.score

cat /opt/project/translation/mbart-nmt/src/hypothesis/${EXP_NAME}_lc_hypothesis.score
