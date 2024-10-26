# build prompt.scp file using feats.scp and utt2prompt. Just a two-stage indexing procedure

import sys

utt2prompt = sys.argv[1]
feats_scp = sys.argv[2]
# prompt_scp = sys.argv[3]

utt2feat = dict()
with open(feats_scp, 'r') as fr:
    for line in fr.readlines():
        terms = line.strip().split()
        utt2feat[terms[0]] = terms[1]

# with open(prompt_scp, 'w') as fw:
with open(utt2prompt, 'r') as fr:
    for line in fr.readlines():
        terms = line.strip().split()
        utt, prompt_utt = terms
        feature_line = utt2feat[prompt_utt]
        # fw.write(f"{utt} {feature_line}\n")
        print(f"{utt} {feature_line}")

