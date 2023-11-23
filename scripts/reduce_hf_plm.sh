BASE_PATH=/opt/project/mbart-nmt

python ${BASE_PATH}/reduce_hf_plm.py --base_path ${BASE_PATH} --plm_name facebook/mbart-large-50-many-to-many-mmt --plm_local_path ${BASE_PATH}/src/plm/tmp_ckpt --reduction_path ${BASE_PATH}/src/plm/reduced_hf_mbart50_m2m --use_pretrained_spm --max_length 512
