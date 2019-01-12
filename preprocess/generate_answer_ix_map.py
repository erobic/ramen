import cPickle
import os
import json


def generate_answer_ix_map(dir, split):
    with open(os.path.join(dir, split + '_ans2label.pkl'), 'rb') as f:
        ans2label = cPickle.load(f)

    with open(os.path.join(dir, split + '_label2ans.pkl'), 'rb') as f:
        label2ans = cPickle.load(f)

    label2ansDict = {}
    for ix, ans in enumerate(label2ans):
        label2ansDict[ix] = ans

    with open(os.path.join(dir, 'answer_ix_map.json'), 'w') as f:
        json.dump({
            'answer_to_ix': ans2label,
            'ix_to_answer': label2ansDict
        }, f)


if __name__ == "__main__":
    generate_answer_ix_map('/hdd/robik/VQA2/bottom-up-attention', 'trainval')
