#!/bin/bash
RAWDIR=/opt/project/translation/repo/mbart-nmt/src/aihub_corpus
SAVEDIR=/opt/project/translation/repo/mbart-nmt/src/pretokenized_aihub_corpus
FN_HEAD=aihub_written_lower
KO_SPLITDIR=ko_split_corpus
KO_TOKDIR=pretokenized_ko_split_corpus
EN_SPLITDIR=en_split_corpus
EN_TOKDIR=pretokenized_en_split_corpus

rm -rf ${SAVEDIR}/${KO_SPLITDIR}/
rm -rf ${SAVEDIR}/${KO_TOKDIR}/
rm -rf ${SAVEDIR}/${EN_SPLITDIR}
rm -rf ${SAVEDIR}/${EN_TOKDIR}/
mkdir ${SAVEDIR}/${KO_SPLITDIR}
mkdir ${SAVEDIR}/${KO_TOKDIR}
mkdir ${SAVEDIR}/${EN_SPLITDIR}
mkdir ${SAVEDIR}/${EN_TOKDIR}

split -a 4 -l 5000 -d ${RAWDIR}/${FN_HEAD}.ko ${SAVEDIR}/${KO_SPLITDIR}/${FN_HEAD}.ko_

split -a 4 -l 5000 -d ${RAWDIR}/${FN_HEAD}.en ${SAVEDIR}/${EN_SPLITDIR}/${FN_HEAD}.en_

python3 pretokenize.py --tagger mecab --input_dir ${SAVEDIR}/${KO_SPLITDIR} --output_dir ${SAVEDIR}/${KO_TOKDIR} --num_processes 48
python3 pretokenize.py --tagger nltk --input_dir ${SAVEDIR}/${EN_SPLITDIR} --output_dir ${SAVEDIR}/${EN_TOKDIR} --num_processes 48

cat ${SAVEDIR}/${KO_TOKDIR}/*.ko_* > ${SAVEDIR}/${FN_HEAD}_pretok.ko
cat ${SAVEDIR}/${EN_TOKDIR}/*.en_* > ${SAVEDIR}/${FN_HEAD}_pretok.en
cat ${SAVEDIR}/${FN_HEAD}_pretok.* > ${SAVEDIR}/${FN_HEAD}_pretok.mono

rm -rf ${SAVEDIR}/${KO_SPLITDIR}
rm -rf ${SAVEDIR}/${KO_TOKDIR}
rm -rf ${SAVEDIR}/${EN_SPLITDIR}
rm -rf ${SAVEDIR}/${EN_TOKDIR}

wc -l ${SAVEDIR}/${DESTDIR}/*
