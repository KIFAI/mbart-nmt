import os, shutil
import sentencepiece.sentencepiece_model_pb2 as model

from tqdm import tqdm
from transformers import MBart50TokenizerFast

#Load pre-trained sentencepiece model
PRETRAINED_SPC_PATH, NEW_VOCAB_PATH, JOINT_SPC_PATH, TMP_DIR = "ko_en/spiece.model", "vi_id_km_hi/vi_id_km_hi.vocab", "joint", "tmp"

m = model.ModelProto()
with open(PRETRAINED_SPC_PATH, "rb") as f:
    pretrained_spc_model = f.read()
m.ParseFromString(pretrained_spc_model)

#Load tokens want to add
token_list = []
with open(NEW_VOCAB_PATH, 'r') as f:
    for l in f:
        token_list.append(l.rstrip())

#Add new tokens to sentencepiece model
added_num = 0
pretrained_vocab = {p.piece:i for i, p in enumerate(m.pieces)}
print(f"pretrained vocab size : {len(m.pieces)}")
for token in tqdm(token_list, total=len(token_list)):
    new_token = model.ModelProto().SentencePiece()
    new_token.piece, new_token.score = token.split('\t')[0], float(token.split('\t')[-1])
    try:
        pretrained_vocab[new_token.piece]
    except:
        m.pieces.append(new_token)
        added_num += 1
print(f"added vocab size : {len(m.pieces)}")


#Save new sentencepiece model
if not os.path.isdir(JOINT_SPC_PATH):
    os.mkdir(JOINT_SPC_PATH)
with open(f"{os.path.join(JOINT_SPC_PATH, 'joint.model')}", "wb") as f:
    f.write(m.SerializeToString())

tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
tokenizer.save_pretrained(TMP_DIR)
os.remove(f"{TMP_DIR}/sentencepiece.bpe.model")
os.remove(f"{TMP_DIR}/tokenizer.json")
shutil.copy(f"{os.path.join(JOINT_SPC_PATH, 'joint.model')}", f"{TMP_DIR}/sentencepiece.bpe.model")

tokenizer = MBart50TokenizerFast.from_pretrained(TMP_DIR)
tokenizer.save_pretrained(JOINT_SPC_PATH)

print(f"added vocab size : {tokenizer.vocab_size}")
