import numpy as np
import torch
import os
import csv
import json
import copy
import torch.nn as nn
from six.moves import cPickle

class VqaUtils:

    @staticmethod
    def normalize_features(curr_image_features):
        norm = np.linalg.norm(curr_image_features, axis=1)
        denom = np.repeat(norm, curr_image_features.shape[1]).reshape(
            (curr_image_features.shape[0], curr_image_features.shape[1]))
        curr_image_features = np.divide(curr_image_features, denom)
        return curr_image_features

    @staticmethod
    def get_linear_features(curr_spatial_features, num_objects, spatial_feature_length):
        linear_features_x, linear_features_y = [], []
        for obj_ix in range(num_objects):
            x_start, x_end = curr_spatial_features[obj_ix][0], curr_spatial_features[obj_ix][2]
            y_start, y_end = curr_spatial_features[obj_ix][1], curr_spatial_features[obj_ix][3]
            curr_feats_x = np.linspace(x_start, x_end, num=spatial_feature_length)
            curr_feats_y = np.linspace(y_start, y_end, num=spatial_feature_length)
            linear_features_x.append(curr_feats_x.tolist())
            linear_features_y.append(curr_feats_y.tolist())

        linear_features_x, linear_features_y = np.array(linear_features_x), np.array(linear_features_y)
        return linear_features_x, linear_features_y

    @staticmethod
    def get_image_features(curr_image_features, curr_spatial_features, spatial_feature_type,
                           spatial_feature_length, num_objects, do_not_normalize_image_feats):
        assert spatial_feature_type is None or spatial_feature_type in ['simple', 'linear',
                                                                        'mesh', 'none'], "Unsupported spatial_feature_type {}".format(
            spatial_feature_type)
        if not do_not_normalize_image_feats:
            curr_image_features = VqaUtils.normalize_features(curr_image_features)
        if spatial_feature_type == 'none':
            return curr_image_features

        if spatial_feature_type == 'simple':
            curr_spatial_features = VqaUtils.normalize_features(curr_spatial_features)
            curr_entry = np.concatenate((curr_image_features, curr_spatial_features), axis=1)
        elif spatial_feature_type == 'linear':
            linear_features_x, linear_features_y = VqaUtils.get_linear_features(curr_spatial_features, num_objects,
                                                                                spatial_feature_length)
            linear_features_x = VqaUtils.normalize_features(linear_features_x)
            linear_features_y = VqaUtils.normalize_features(linear_features_y)
            curr_entry = np.concatenate((curr_image_features, linear_features_x, linear_features_y), axis=1)
        elif spatial_feature_type == 'mesh':
            linear_features_x, linear_features_y = VqaUtils.get_linear_features(curr_spatial_features, num_objects,
                                                                                spatial_feature_length)
            meshes = []
            for obj_ix in range(num_objects):
                curr_mesh = np.array(np.meshgrid(linear_features_x[obj_ix], linear_features_y[obj_ix])).flatten()
                meshes.append(curr_mesh)
            if not do_not_normalize_image_feats:
                meshes = VqaUtils.normalize_features(np.array(meshes))
            curr_entry = np.concatenate((curr_image_features, meshes), axis=1)
        else:
            curr_entry = VqaUtils.normalize_features(curr_image_features)

        return curr_entry

    @staticmethod
    def save_stats(stats, per_type_metrics, all_preds, save_dir, split, epoch):
        VqaUtils.save_csv(stats, save_dir, stats_filename='overall_stats.csv')
        VqaUtils.save_csv(per_type_metrics, save_dir, stats_filename='per_type_stats.csv')
        with open(os.path.join(save_dir, 'overall_stats.json'), 'w') as of:
            json.dump(stats, of)
        with open(os.path.join(save_dir, 'per_type_stats.json'), 'w') as pf:
            json.dump(per_type_metrics, pf)
        pred_dir = os.path.join(save_dir, 'predictions')
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        with open(os.path.join(pred_dir, 'prediction_{}_epoch_{}.json'.format(split, epoch)), 'w') as pred_f:
            json.dump(all_preds, pred_f)

    @staticmethod
    def save_preds(all_preds, save_dir, split, epoch):
        pred_dir = os.path.join(save_dir, 'predictions')
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        with open(os.path.join(pred_dir, 'prediction_{}_epoch_{}.json'.format(split, epoch)), 'w') as pred_f:
            json.dump(all_preds, pred_f)

    @staticmethod
    def save_csv(stats, save_dir, stats_filename='overall_stats.csv'):
        FIELDS = stats[0].keys()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, stats_filename), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for stat in stats:
                writer.writerow(stat)

    @staticmethod
    def get_spatial_length(spatial_feature_type, spatial_feature_length):
        if spatial_feature_type == 'none':
            return 0
        elif spatial_feature_type == 'mesh':
            return 2 * spatial_feature_length * spatial_feature_length
        elif spatial_feature_type == 'linear':
            return 2 * spatial_feature_length
        elif spatial_feature_type == 'simple':
            return 4
        else:
            return 0


    @staticmethod
    def get_question_type(full_question, use_clevr_style=False):
        if use_clevr_style:
            if 'program' in full_question:
                question_type = full_question['program'][-1]['function']
            else:
                question_type = "unknown"
        else:
            if 'vqa_annotation' in full_question:
                question_type = full_question['vqa_annotation']['question_type']
            else:
                print("full_question {}".format(full_question))
                question_type = "unknown"
        return question_type


class PerTypeMetric:
    def __init__(self, epoch):
        self.epoch = epoch
        self.per_type_correct = {}
        self.per_type_total = {}
        self.per_type_acc = {}

    def update_with_pred(self, full_question, label, output,use_clevr_style):
        if use_clevr_style:
            question_type = full_question['program'][-1]['function']
        else:
            question_type = full_question['vqa_annotation']['question_type']
        self.update_for_question_type(question_type, label, output)

    def update_for_question_type(self, question_type, label, output):
        # For CLEVR, the score would be 1 if correct else 0. For VQA2, it would be soft scores which 0 for incorrect answers.
        max_ix = np.argmax(output)
        self.update(question_type, label[max_ix])

    def update(self, question_type, score):
        if question_type not in self.per_type_correct:
            self.per_type_correct[question_type] = 0
        if question_type not in self.per_type_total:
            self.per_type_total[question_type] = 0

        # if is_correct:
        # if score > 0:
        #     print("score {}".format(score))
        self.per_type_correct[question_type] += score
        self.per_type_total[question_type] += 1

        self.per_type_acc[question_type] = self.per_type_correct[question_type] / self.per_type_total[question_type]

    def to_string(self):
        # print(self.per_type_acc)
        return self.per_type_acc

    def get_json(self):
        data = copy.deepcopy(self.per_type_acc)
        data['epoch'] = self.epoch
        return data


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def load_answer_scores(dataroot, split):
    question_path = os.path.join(dataroot, 'questions', '%s_questions.json' %split)
    questions = sorted(json.load(open(question_path))['questions'], key=lambda x:x['question_id'])
    entries = {}
    answer_path = os.path.join(dataroot, 'features', '%s_target.pkl' % split)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    assert (len(questions) == len(answers))
    for question, answer in zip(questions, answers):
        assert (question['question_id'] == answer['question_id'])
        entries[answer['question_id']] = answer
    return entries


def normalize(curr_image_features):
    norm = np.linalg.norm(curr_image_features, axis=1)
    denom = np.repeat(norm, curr_image_features.shape[1]).reshape(
        (curr_image_features.shape[0], curr_image_features.shape[1]))
    curr_image_features = np.divide(curr_image_features, denom)
    return curr_image_features


if __name__ == "__main__":
    load_answer_scores('/hdd/robik/CVQA', split='train')
