import json
import os

# Adds dummy answers for 'test' questions into 'test_dev' predictions because evalAI requires
# all questions to be answered
test_dev_qids = {}
if __name__ == "__main__":
    path = '/hdd/robik/VQA2_results/Ramen_VQA2/predictions'
    test_dev_preds = json.load(open(path + '/prediction_test_dev_epoch_15.json'))

    test_dev_questions = json.load(open('/hdd/robik/VQA2/questions/test_dev_questions.json'))
    for q in test_dev_questions['questions']:
        test_dev_qids[q['question_id']] = 1

    test_questions = json.load(open('/hdd/robik/VQA2/questions/test_questions.json'))

    for q in test_questions['questions']:
        if q['question_id'] not in test_dev_qids:
            test_dev_preds.append({'question_id': int(q['question_id']), 'answer': 'dummy_answer'})
    json.dump(test_dev_preds,
              open(os.path.join(path + '/prediction_test_dev_epoch_15_complete.json'), 'w'))
