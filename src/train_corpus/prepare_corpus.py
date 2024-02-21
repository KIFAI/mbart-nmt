import os, glob
import pandas as pd

COL2LID = {'ko': 'ko_KR', 'en':'en_XX', 'vi':'vi_VN', 'id': 'id_ID', 'km':'km_KH', 'hi': 'hi_IN'}

TRAIN_RATIO = 0.999

LOAD_DIR = '/opt/project/mbart-nmt/src/train_corpus/total_corpus'
SUB_DIRS = [el.split('/')[-1] for el in list(filter(os.path.isdir, [os.path.join(LOAD_DIR, fn) for fn in os.listdir(LOAD_DIR)])) if len(el.split('/')[-1]) == 5]
print(f"SUB_DIRS : {SUB_DIRS}")
SAVE_DIR = '/opt/project/mbart-nmt/src/train_corpus/cased_corpus_exp'

for sub_dir in SUB_DIRS:

    files = glob.glob(f"{os.path.join(LOAD_DIR, sub_dir)}/*")

    tmp = pd.DataFrame()

    for f in files:
        corpus_df = pd.read_csv(f, sep="\t")
        src_col, tgt_col = sub_dir.split('2')
        assert list(corpus_df.columns) == ["domain", "subdomain", src_col, tgt_col]
        tmp = pd.concat([tmp, corpus_df], axis=0)
    print(f"\nMerged shape of {sub_dir}: {tmp.shape[0]}")

    tmp.columns = ["domain", "subdomain", COL2LID[src_col], COL2LID[tgt_col]]

    total = tmp.drop_duplicates([COL2LID[src_col], COL2LID[tgt_col]], keep='first').reset_index(drop=True)
    print(f"total df's shape after dropping duplicated cases : {total.shape[0]}")

    train = total.sample(frac=TRAIN_RATIO, replace=False, axis=0, random_state=0)
    train.to_csv(os.path.join(SAVE_DIR, f"train_corpus_{sub_dir}_{train.shape[0]}.tsv"), sep="\t", encoding="utf-8-sig", index=False)

    valid = total[~total.index.isin(train.index)]
    valid.to_csv(os.path.join(SAVE_DIR, f"valid_corpus_{sub_dir}_{valid.shape[0]}.tsv"), sep="\t", encoding="utf-8-sig", index=False)
