BASE=/opt/project/translation/repo/mbart-nmt
RAW=${BASE}/src/aihub_corpus

MONO_FILE=${BASE}/src/pretokenized_aihub_corpus/aihub_written_pretok.mono
OUT_FN=upper_aihub
python train_sentencepiece.py --mono_file ${MONO_FILE} --out_fn ${OUT_FN}

SPM=${BASE}/sentencepiece/build/src/spm_encode
MONO_MODEL=${BASE}/prepare_corpus/spc_model/mono.${OUT_FN}.model
TRAIN_DATA=${BASE}/src/train_corpus/spm_pretok_upper_aihub_corpus
rm -rf ${TRAIN_DATA}
mkdir ${TRAIN_DATA}

TRAIN=train
VALID=valid
TEST=test
SRC=en
TGT=ko
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${TRAIN}.${SRC} > ${TRAIN_DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${TRAIN}.${TGT} > ${TRAIN_DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${VALID}.${SRC} > ${TRAIN_DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${VALID}.${TGT} > ${TRAIN_DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${TEST}.${SRC} > ${TRAIN_DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MONO_MODEL} < ${RAW}/aihub_written_${TEST}.${TGT} > ${TRAIN_DATA}/${TEST}.spm.${TGT} &
