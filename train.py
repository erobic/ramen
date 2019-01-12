import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tdiuc import mean_per_class
import numpy as np
import json
from tqdm import tqdm
import sys
import time
from vqa_utils import VqaUtils, PerTypeMetric
from tdiuc import tdiuc_mean_per_class


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


def train(model, train_loader, eval_loader, num_epochs, opt, args, start_epoch=0, best_val_score=0, best_epoch=0):
    all_stats = []
    test_per_type_metrics = []
    if not args.test or args.test_has_answers:
        with open(os.path.join(args.data_root, 'vqa2', 'val_annotations.json')) as f:
            val_gt_annotations = json.load(f)
    else:
        val_gt_annotations = None

    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters())) \
        if opt is None else opt

    for epoch in range(start_epoch, num_epochs):
        is_best = False
        start = time.time()
        train_loss = 0
        train_raw_score = 0
        train_score = 0
        train_normalized_score = 0
        train_end = 0
        total_norm = 0
        count_norm = 0
        if not args.test:
            train_N = len(train_loader.dataset)
        else:
            train_N = 1
        running_N = 0
        train_upper_bound = 0

        if not args.test:
            for i, (v, b, q, a, qt, qid) in enumerate(train_loader):
                v = Variable(v.float()).cuda()
                b = Variable(b.float()).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()

                ub = a.max(1)[0].sum()
                train_upper_bound += ub

                pred = model(v, b, q, a)
                loss = instance_bce_with_logits(pred, a)
                loss.backward()
                total_norm += nn.utils.clip_grad_norm(model.parameters(), 0.25)
                count_norm += 1
                optimizer.step()
                optimizer.zero_grad()

                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_loss += loss.data[0] * v.size(0)
                train_raw_score += batch_score
                curr_size = v.shape[0]
                running_N += curr_size
                if i % 100 == 0:
                    print(
                        "Epoch {} Loss {} Total Score {:.2f}"
                            .format(epoch, train_loss / running_N,
                                    100 * train_raw_score / running_N))

            train_end = time.time()
            train_loss /= train_N
            train_normalized_score = 100 * train_raw_score / train_upper_bound
            train_score = 100 * train_raw_score / train_N

        if None != eval_loader:
            model.train(False)
            val_raw_score, val_upper_bound, val_N, val_loss, all_preds, test_per_type_metric, tdiuc_results \
                = evaluate(model, eval_loader, epoch, args, gt_annotations=val_gt_annotations)
            if not args.test or args.test_has_answers:
                val_loss /= val_N
                val_normalized_score = 100 * val_raw_score / val_upper_bound
                val_score = 100 * val_raw_score / val_N
            model.train(True)

        if not args.test or args.test_has_answers:
            if eval_loader is not None:
                print("test_per_type_metric {}".format(json.dumps(test_per_type_metric.get_json(), indent=4)))

            val_end = time.time()
            test_per_type_metrics.append(test_per_type_metric.get_json())

            if eval_loader is not None and val_score > best_val_score:
                is_best = True
                print("Best val score {} at epoch {}".format(best_val_score, epoch))
                best_val_score = val_score
                best_epoch = epoch

            stats = {
                "train_loss": float(train_loss),
                "train_raw_score": float(train_raw_score / train_N),
                "train_normalized_score": float(train_normalized_score),
                "train_upper_bound": float(train_upper_bound / train_N),
                "train_score": float(train_score),
                "train_N": train_N,

                "val_raw_score": float(val_raw_score),
                "val_normalized_score": float(val_normalized_score),
                "val_upper_bound": float(val_upper_bound/val_N),
                "val_loss": float(val_loss),
                "val_score": float(val_score),
                "val_N": val_N,

                "best_val_score": float(best_val_score),
                "best_epoch": best_epoch,

                "epoch": epoch,
                "train_time": train_end - start,
                "val_time": val_end - train_end,
                "test_per_type_metric": test_per_type_metric.get_json()
            }
            print(json.dumps(stats, indent=4))

            save_data = stats.copy()
            save_data["model_state_dict"] = model.state_dict()
            save_data["optimizer_state_dict"] = optimizer.state_dict()
            save_data["args"] = args
            if tdiuc_results is not None:
                stats["tdiuc_results"] = tdiuc_results
            with open(os.path.join(args.expt_save_dir, 'latest-model.pth'), 'wb') as lmf:
                torch.save(save_data, lmf)
            all_stats.append(stats)
            VqaUtils.save_stats(all_stats, test_per_type_metrics, all_preds, tdiuc_results, args.expt_save_dir,
                                split='val', epoch=epoch)
            if is_best:
                with open(os.path.join(args.expt_save_dir, 'best-model.pth'), 'wb') as bmf:
                    torch.save(save_data, bmf)

        if args.test:
            VqaUtils.save_preds(all_preds, args.expt_save_dir, args.test_split, epoch)
            print("Test completed!")
            break


def evaluate(model, dataloader, epoch, args, gt_annotations):
    is_tdiuc = args.data_set.lower() == 'tdiuc'
    score = 0
    upper_bound = 0
    num_data = 0
    total_loss = 0
    per_type_metric = PerTypeMetric(epoch=epoch, dataset=args.dataset)
    with open(os.path.join(args.data_root, 'bottom-up-attention', 'answer_ix_map.json')) as f:
        answer_ix_map = json.load(f)

    all_preds = []

    for v, b, q, a, qt, qid in iter(dataloader):
        v = Variable(v.float()).cuda()
        b = Variable(b.float()).cuda()
        q = Variable(q, volatile=True).cuda()

        if not args.test or args.test_has_answers:
            a = a.cuda()

        pred = model(v, b, q, None)

        if not args.test or args.test_has_answers:
            batch_score = compute_score_with_logits(pred, a).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

            loss = instance_bce_with_logits(pred, a)
            total_loss += loss.data[0] * v.size(0)

        pred_ans_ixs = pred.max(1)[1]
        for curr_ix, pred_ans_ix in enumerate(pred_ans_ixs):
            pred_ans = answer_ix_map['ix_to_answer'][str(int(pred_ans_ix))]
            all_preds.append({
                'question_id': int(qid[curr_ix].data),
                'answer': str(pred_ans)
            })
            if not args.test or args.test_has_answers:
                per_type_metric.update_for_question_type(qt[curr_ix], a[curr_ix].cpu().data.numpy(),
                                                         pred[curr_ix].cpu().data.numpy())
    if is_tdiuc and (not args.test or args.test_has_answers):
        predictions_np = np.array(all_preds)
        tdiuc_results = tdiuc_mean_per_class(predictions_np, gt_annotations['annotations'],
                                             answer_ix_map['answer_to_ix'])
        print("TDIUC metrics {}".format(tdiuc_results))
    else:
        tdiuc_results = None

    return score, upper_bound, len(dataloader.dataset), total_loss, all_preds, per_type_metric, tdiuc_results
