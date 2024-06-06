
# imports modules for registration

from datasets import load_dataset
from tqdm import tqdm
import os
import difflib

auth_token = os.environ["HF_TOKEN"]  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

import difflib


for example in winoground:
    sentence1 = example["caption_0"].lower().split(" ")
    sentence2 = example["caption_1"].lower().split(" ")

    diff = []
    diff2 = []
    union = set()
    for i, word in enumerate(sentence1):
        if sentence2[i] != word:
            if sentence2[i] not in union and word not in union:
                diff.append(word)
                diff2.append(sentence2[i])
                union.add(word)
                union.add(sentence2[i])

    print(' '.join(diff), ' '.join(diff2))