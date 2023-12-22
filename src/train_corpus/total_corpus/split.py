import pandas as pd

total = pd.read_csv("total_corpus.tsv", sep="\t")
print(f"total df's shape : {total.shape[0]}")
total = total.drop_duplicates(["ko_KR", "en_XX"], keep="first").reset_index(drop=True)
print(f"total df's shape after dropping duplicated cases : {total.shape[0]}")

train = total.sample(frac=0.9998, replace=False, axis=0, random_state=0)
train.to_csv(f"train_corpus_{train.shape[0]}.tsv", sep="\t", encoding="utf-8-sig", index=False)

valid = total[~total.index.isin(train.index)]

valid.to_csv(f"valid_corpus_{valid.shape[0]}.tsv", sep="\t", encoding="utf-8-sig", index=False)

print(f"train + valid shape : {train.shape[0] + valid.shape[0]}")
print(f"total shape : {total.shape[0]}")
