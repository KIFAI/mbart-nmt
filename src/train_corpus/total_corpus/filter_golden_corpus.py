import glob
import pandas as pd

train_fn = glob.glob('train_corpus_*.tsv')
if len(train_fn) != 1:
    raise ValueError("Remove unncessary train corpus")

train_df = pd.read_csv(train_fn[0], sep='\t')
golden_df = pd.read_csv('../golden_corpus/total.tsv', sep='\t')

train_ko_dict = {v:k for k, v in train_df['ko_KR'].to_dict().items()}
golden_ko_dict = {v:k for k, v in golden_df['ko_KR'].to_dict().items()}

check_idxs = []
for ko in golden_ko_dict.keys():
    try:
        check_idxs.append(train_ko_dict[ko])
    except:
        pass

train_filtered_df = train_df.drop(train_df.index[check_idxs])
assert (train_df.shape[0] - train_filtered_df.shape[0]) == len(check_idxs)

train_filtered_df.to_csv(f"train_corpus_filtered_{train_filtered_df.shape[0]}.tsv", sep='\t', encoding='utf-8-sig', index=False)
print('Successfully filtered..')
