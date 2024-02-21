import os, glob
import pandas as pd
import matplotlib.pyplot as plt

corpus_dirs = [c for c in list(filter(os.path.isdir, os.listdir('./'))) if 'en' not in c and 'statics' != c]
print(corpus_dirs)

for corpus_dir in corpus_dirs:
    merged = pd.DataFrame()
    files = glob.glob(f"{corpus_dir}/*")

    for fn in files:
        merged = pd.concat([merged, pd.read_csv(fn, sep='\t')], axis=0)

    src_lang, tgt_lang = corpus_dir.split('2')
    print(f"The shape of merged {tgt_lang} df is {merged.shape}")

    merged['token_length'] = merged[tgt_lang].apply(lambda x: len(x.split()))

    quantiles = merged['token_length'].quantile([0.25, 0.5, 0.75])

    plt.figure(figsize=(8, 5))
    plt.hist(merged['token_length'], bins=20, color='skyblue', edgecolor='black')
    for q in quantiles:
        plt.axvline(q, color='red', linestyle='--')
    plt.title('Distribution of Hi Token Length')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.legend(['25th', '50th', '75th', 'Data'])
    plt.savefig(f'statics/{tgt_lang}_token_length_distribution.png')
    plt.show()

    domain_stats = merged.groupby('domain')['token_length'].describe().reset_index()
    domain_stats.to_csv(f'statics/{tgt_lang}_statics.csv')

    print(f"\nDomain-wise Statistics for {tgt_lang} Token Length:\n")
    print(domain_stats)
