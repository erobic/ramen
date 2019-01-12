from __future__ import division
import json
import csv
import numpy as np
from collections import defaultdict
from scipy import stats
import sys, argparse
import json
import sys
py_version = sys.version_info[0 ]


def load_json(json_preds, gt_ann_a, answerkey):
    preds_json = json.load(open(json_preds))
    return preds_json
    # return preds_to_answer_ixs(preds_json, gt_ann_a, answerkey)

#
# def preds_to_answer_ixs(preds_json, gt_ann_a, answerkey):
#     lut_preds = dict()
#     for pred in preds_json:
#         lut_preds[pred['question_id']] = pred['answer']
#     predictions = []
#     for idx, ans in enumerate(gt_ann_a):
#         if idx % 1000 == 0:
#             sys.stdout.write('\rAligning.. ' + str(idx))
#             sys.stdout.flush()
#         try:
#             predictions.append(lut_preds[ans['question_id']])
#         except:
#             print("THIS SHOULD NOT HAPPEN")
#     predictions = [int(answerkey[p]) for p in predictions]
#     return predictions


def tdiuc_mean_per_class(predictions, gt_ann, answerkey, convert_preds=True):
    return mean_per_class(predictions, gt_ann, answerkey, convert_preds)


def get_gt_ann_map(gt_ann):
    gt_ann_map = {}
    for ann in gt_ann:
        gt_ann_map[ann['question_id']] = ann
    return gt_ann_map


def mean_per_class(predictions, overall_gt_ann, answerkey, convert_preds=True):
    # predictions = preds_to_answer_ixs(predictions, gt_ann, answerkey)
    res_json = {}
    res = defaultdict(list)
    pred_answer_list = []
    gt_answer_list = []
    notfound = 0
    gt_ann_map = get_gt_ann_map(overall_gt_ann)
    for idx, pred in enumerate(predictions):
        qid = pred['question_id']
        pred_ans = pred['answer']
        qid_gt_ann = gt_ann_map[qid]

        gt_answer = qid_gt_ann['answers'][0]['answer']
        gt_type = qid_gt_ann['question_type']
        res[gt_type + '_pred'].append(pred_ans)
        if py_version == 2:
            gt_answer_present =answerkey.has_key(gt_answer)
        else:
            gt_answer_present = gt_answer in answerkey
        if gt_answer_present:
            #gt_idx = int(answerkey[gt_answer])
            #res[gt_type + '_gt'].append(gt_idx)
            res[gt_type + '_gt'].append(gt_answer)
            # gt_answers_idx.append(gt_idx)
            pred_answer_list.append(pred_ans)
            gt_answer_list.append(gt_answer)
            # if gt_idx == pred:

            if gt_answer == pred_ans:
                res[gt_type + '_t'].append(pred_ans)
            else:

                res[gt_type + '_f'].append(pred_ans)
        else:
            gt_answer_list.append(-1)
            pred_answer_list.append(pred_ans)
            res[gt_type + '_f'].append(pred_ans)
            res[gt_type + '_gt'].append(-1)
            notfound += 1
    print("\n %d of validation answers were not in the answerkey" % notfound)
    types = list(set([a['question_type'] for a in overall_gt_ann]))
    sum_acc = []
    eps = 1e-10
    # print('\nNOT USING PER-ANSWER NORMALIZATION\n')
    res_json['without_normalization'] = []
    for tp in types:
        denom = len(res[tp + '_t'] + res[tp + '_f'])
        if denom == 0:
            acc = 0
        else:
            acc = 100 * (len(res[tp + '_t']) / denom)
        sum_acc.append(acc + eps)
        # print('Accuracy for %s is %.2f' % (tp, acc))
        res_json['without_normalization'].append({'type': tp, 'accuracy': acc})

    res_json['arithmetic_mpt'] = np.mean(np.array(sum_acc))
    # print('Arithmetic MPT Accuracy is %.2f' % (np.mean(np.array(sum_acc))))

    res_json['harmonic_mpt'] = stats.hmean(sum_acc)
    # print('Harmonic MPT Accuracy is %.2f' % (stats.hmean(sum_acc)))

    for pr in predictions:
        gt_ann_map[pr['question_id']]['answers'][0]

    n_acc = 100 * np.mean(np.array(pred_answer_list) == np.array(gt_answer_list))
    res_json['overall_without_normalization'] = n_acc
    # print('Overall Traditional Accuracy is %.2f' % n_acc)
    print('\n---------------------------------------')
    res_json['with_normalization'] = []
    # print('USING PER-ANSWER NORMALIZATION\n')
    types = list(set([a['question_type'] for a in overall_gt_ann]))
    sum_acc = []
    eps = 1e-10
    for tp in types:
        per_ans_stat = defaultdict(int)
        for g, p in zip(res[tp + '_gt'], res[tp + '_pred']):
            per_ans_stat[str(g) + '_gt'] += 1
            if g == p:
                per_ans_stat[str(g)] += 1
        unq_acc = 0
        for unq_ans in set(res[tp + '_gt']):
            acc_curr_ans = per_ans_stat[str(unq_ans)] / per_ans_stat[str(unq_ans) + '_gt']
            unq_acc += acc_curr_ans
        denom = len(set(res[tp + '_gt']))
        if denom == 0:
            acc = 0
        else:
            acc = 100 * unq_acc / denom
        sum_acc.append(acc + eps)
        # print('Accuracy for %s is %.2f' % (tp, acc))
        res_json['with_normalization'].append({
            'type': tp,
            'accuracy': acc
        })

    res_json['arithmetic_nmpt'] = np.mean(np.array(sum_acc))
    # print('Arithmetic N-MPT Accuracy is %.2f' % (np.mean(np.array(sum_acc))))
    res_json['harmonic_nmpt'] = stats.hmean(sum_acc)
    # print('Harmonic N-MPT Accuracy is %.2f' % (stats.hmean(sum_acc)))
    n_acc = 100 * np.mean(np.array(pred_answer_list) == np.array(gt_answer_list))
    res_json['overall_with_normalization'] = n_acc
    # print('Overall Traditional Accuracy is %.2f' % n_acc)

    return res_json


def generate_answer_key(annotations):
    answer_ix = 0
    answer_key = {}
    for ann in annotations['annotations']:
        ans = ann['answers'][0]['answer']
        if ans not in answer_key:
            answer_key[ans] = answer_ix
            answer_ix += 1


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_ann', required=True,
                        help='path to ground truth annotation JSON file')
    parser.add_argument('--pred_ann', required=True,
                        help='path to the predictions JSON file')
    parser.add_argument('--answer_ix_map', required=True,
                        help='Json file with ix_to_answer for each answer')
    args = parser.parse_args()
    with open(args.answer_ix_map, 'rt') as f:
        answer_ix_map = json.load(f)['ix_to_answer']
        answerkey = dict((ans, int(ix)) for ix, ans in zip(answer_ix_map.keys(), answer_ix_map.values()))

    # answerkey_csv = csv.reader(open(args.answerkey))
    # answerkey = dict((rows[0], rows[1]) for rows in answerkey_csv)
    gt_ann = json.load(open(args.gt_ann))['annotations']
    predictions = load_json(args.pred_ann, gt_ann, answerkey)
    predictions = np.array(predictions)
    mean_per_class(predictions, gt_ann, answerkey, convert_preds=False)
    print('.------------------------------------------')


# %%
if __name__ == "__main__":
    main()
