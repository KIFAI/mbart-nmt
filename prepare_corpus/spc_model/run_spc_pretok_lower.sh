BASE=/opt/project/translation/repo/mbart-nmt/src
IN_FN=aihub_written_lower_pretok.mono
OUT_FN=mono.lower_aihub
python prepare_sentencepiece.py --corpus_path ${BASE}/pretokenized_aihub_corpus --spc_path ${BASE}/sentencepiece --input_fn ${IN_FN} --out_fn ${OUT_FN}

