import glob
import pandas as pd

files = glob.glob("./total/*.tsv")
print(files)
tmp = pd.DataFrame()
columns = ["domain", "subdomain", "ko", "en"]
for f in files:
    corpus_df = pd.read_csv(f, sep="\t")
    if not list(corpus_df.columns) == columns:
        corpus_df = corpus_df[columns]
    tmp = pd.concat([tmp, corpus_df], axis=0)

assert list(tmp.columns) == columns

print(f"Merged shape: {tmp.shape}")

print(f"dropna shape : {tmp.dropna().shape}")
tmp.columns = ["domain", "subdomain", "ko_KR", "en_XX"]
print(tmp.head())
tmp.dropna().to_csv("total_corpus.tsv", sep="\t", index=False, encoding="utf-8-sig")
