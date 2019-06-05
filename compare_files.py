import json
from six.moves import cPickle
import os

root = '/hdd/robik/projects/ramen_debug'

def compare_targets():
    working_target = json.load(open(os.path.join(root, 'working', 'train_target.json')))
    not_working_target = json.load(open(os.path.join(root, 'not_working', 'train_target.json')))

    working_target_qns = {}
    for w in working_target:
        working_target_qns[w['question_id']] = w
    not_working_target_qns = {}

    not_working_minus_working = {}
    for nw in not_working_target:
        not_working_target_qns[nw['question_id']] = nw
        qid = nw['question_id']
        if str(qid) not in working_target_qns and int(qid) not in working_target_qns:
            not_working_minus_working[qid] = nw

    total = 0
    present_in_w_but_not_in_nw = 0
    for ix, qid in enumerate(not_working_minus_working):
        if len(not_working_minus_working[qid]['scores']) != 0:
            present_in_w_but_not_in_nw += 1
        total += 1
    print("total {} present_in_w_but_not_in_nw {}".format(total, present_in_w_but_not_in_nw))

    working_minus_not_working = {}
    for w in working_target:
        working_target_qns[w['question_id']] = w
        qid = w['question_id']
        if str(qid) not in not_working_target_qns and int(qid) not in not_working_target_qns:
            working_minus_not_working[qid] = w
    total = 0
    present_in_nw_but_not_in_w = 0
    for ix, qid in enumerate(working_minus_not_working):
        if len(working_minus_not_working[qid]['scores']) != 0:
            present_in_nw_but_not_in_w += 1
        total += 1
    print("total {} present_in_nw_but_not_in_w {}".format(total, present_in_nw_but_not_in_w))


def compare_answer():
    working_answers = json.load(open('/hdd/robik/projects/ramen_debug/working/answer_ix_map.json'))
    not_working_answers = json.load(open('/hdd/robik/projects/ramen_debug/not_working/answer_ix_map.json'))


if __name__ == "__main__":
    compare_targets()
    #compare_answer()