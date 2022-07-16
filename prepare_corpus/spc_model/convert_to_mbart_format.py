edited = []
for line in open("mono.upper_aihub.vocab", 'r', encoding='utf-8'):
    if line in ["<unk>\n", "<s>\n", "</s>\n"]:
        continue
    new_line = line.rstrip('\n') + " 1\n"
    edited.append(new_line)

with open('mono.upper_aihub_mbart.vocab', 'w') as f:
    for e in edited:
        f.write(e)
