BASE_PATH=/opt/project/translation/repo/mbart-nmt

python ${BASE_PATH}/reduce_hf_plm.py --plm_name facebook/mbart-large-50-many-to-many-mmt --plm_local_path ${BASE_PATH}/src/plm/tmp_ckpt --reduction_path ${BASE_PATH}/src/plm/reduced_hf_mbart50_m2m
