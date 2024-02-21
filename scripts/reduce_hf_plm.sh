BASE_PATH=/opt/project/mbart-nmt
PRETRAINED_SPM_PATH=src/sentencepiece/joint/joint.model
REDUCTION_NAME=reduced_hf_mbart06_ko_en_vi_id_km_hi
SUPPORT_LANGS=ko,en,id,km

python ${BASE_PATH}/src/plm/reduce_hf_plm.py --base_path ${BASE_PATH} --plm_name facebook/mbart-large-50-many-to-many-mmt --plm_local_path ${BASE_PATH}/src/plm/tmp_ckpt --reduction_path ${BASE_PATH}/src/plm/${REDUCTION_NAME} --use_pretrained_spm --pretrained_spm_path ${PRETRAINED_SPM_PATH} --support_langs ${SUPPORT_LANGS} --max_length 512
