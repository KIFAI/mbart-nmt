def validate_vocab(ctrans_path):
    with open(f"{ctrans_path}/shared_vocabulary.txt", "r") as f:
        vocab = f.readlines()

    vocab = [s.strip() for s in vocab]

    base_token = ['<s>', '<pad>', '</s>', '<unk>']

    if sum([True for t in base_token if t in vocab]) ==4 :
        print(f"{base_token} tokens are checked in shared vocab")
    else:
        ommition_case = [t for t in base_token if t not in vocab]
        raise ValueError("Check {ommition_case} tokens in vocab")

    base_token_ixs = [i for i, w in enumerate(vocab) if w in base_token]

    if base_token_ixs != [0,1,2,3]:
        print("Change vocab's order normally...")
        for b_t in reversed(base_token):
            print(f"popped (token : {b_t}, ix : {vocab.index(b_t)})")
            vocab.pop(vocab.index(b_t))
        for b_t in reversed(base_token):
            vocab.insert(0, b_t)
        print(f"Vocab's order is changed normally, as like {vocab[:4]}")
        assert vocab[:4] == base_token
    else:
        print("Vocab's order is checked, normally")

    with open(f"{ctrans_path}/shared_vocabulary.txt", "w") as f:
        f.write("\n".join(vocab))
