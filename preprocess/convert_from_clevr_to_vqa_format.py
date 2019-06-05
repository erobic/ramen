import json
import os
import argparse


def convert_from_clevr_to_vqa_format(data_root, split):
    print(f"Converting {split} questions to VQA2 format...")
    clevr_file = json.load(open(os.path.join(data_root, 'questions', f'CLEVR_{split}_questions.json')))
    qns = clevr_file['questions']
    vqa_qns = []
    vqa_annotations = []
    for qn in qns:
        vqa_qn = {
            'question': qn['question'],
            'question_id': qn['question_index'],
            'image_id': qn['image_index']
        }
        vqa_ann = {
            'image_id': qn['image_index'],
            'question_id': qn['question_index']
        }

        if 'test' not in split:
            qtype = qn['program'][-1]['function']
            answer = qn['answer']
            answers = [{'answer': qn['answer']}]
            vqa_ann['answer_type'] = qtype
            vqa_ann['question_type'] = qtype
            vqa_ann['multiple_choice_answer'] = answer
            vqa_ann['answers'] = answers
        vqa_qns.append(vqa_qn)
        vqa_annotations.append(vqa_ann)

    vqa_qns = {'questions': vqa_qns}
    vqa_annotations = {'annotations': vqa_annotations}

    with open(os.path.join(data_root, 'questions', f'{split}_questions.json'), 'w') as f:
        json.dump(vqa_qns, f)
    if 'test' not in split:
        with open(os.path.join(data_root, 'questions', f'{split}_annotations.json'), 'w') as f:
            json.dump(vqa_annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root')
    args = parser.parse_args()
    convert_from_clevr_to_vqa_format(args.data_root, 'train')
    convert_from_clevr_to_vqa_format(args.data_root, 'val')
    convert_from_clevr_to_vqa_format(args.data_root, 'test')
    print("Done with conversions!")
