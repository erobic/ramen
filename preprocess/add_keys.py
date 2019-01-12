import os
import json

if __name__ == "__main__":
    root = '/hdd/robik/CVQA_1000'
    splits = ['train', 'val', 'test']
    for split in splits:
        with open(os.path.join(root, 'questions', '{}_questions.json'.format(split))) as f:
            qns = json.load(f)
            if 'questions' not in qns:
                qns = {'questions': qns}
        with open(os.path.join(root, 'questions', '{}_questions.json'.format(split)), 'w') as f:
            json.dump(qns, f)

        with open(os.path.join(root, 'questions', '{}_annotations.json'.format(split))) as f:
            anns = json.load(f)
            if 'annotations' not in anns:
                anns = {'annotations': anns}
        with open(os.path.join(root, 'questions', '{}_annotations.json'.format(split)), 'w') as f:
            json.dump(anns, f)