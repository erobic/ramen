import json
import os
import time
import torch
import torch.nn as nn

from torch.autograd import Variable

from vqa_utils import VqaUtils, PerTypeMetric
from metrics import Metrics, accumulate_metrics


def instance_bce_with_logits(logits, labels):
    """
    Computes binary cross entropy loss
    :param logits:
    :param labels:
    :return:
    """
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    """
    Computes softscores
    :param logits:
    :param labels:
    :return:
    """
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def save_metrics_n_model(metrics, model, optimizer, args, is_best):
    """
    Saves all the metrics, parameters of models and parameters of optimizer.
    If current score is the highest ever, it also saves this model as the best model
    """
    metrics_n_model = metrics.copy()
    metrics_n_model["model_state_dict"] = model.state_dict()
    metrics_n_model["optimizer_state_dict"] = optimizer.state_dict()
    metrics_n_model["args"] = args

    with open(os.path.join(args.expt_save_dir, 'latest-model.pth'), 'wb') as lmf:
        torch.save(metrics_n_model, lmf)

    if is_best:
        with open(os.path.join(args.expt_save_dir, 'best-model.pth'), 'wb') as bmf:
            torch.save(metrics_n_model, bmf)

    return metrics_n_model


def train(model, train_loader, val_loader, num_epochs, optimizer, args, start_epoch=0, best_val_score=0,
          best_val_epoch=0):
    """
    This is the main training loop. It trains the model, evaluates the model and saves the metrics and predictions.
    """
    metrics_stats_list = []
    val_per_type_metric_list = []

    if optimizer is None:
        # lr_decay_step = 2
        # lr_decay_rate = .25
        # lr_decay_epochs = range(10, 15, lr_decay_step)
        # gradual_warmup_steps = [0.5 * lr, 1.0 * lr, 1.5 * lr, 2.0 * lr]
        optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, model.parameters()),
                                                         lr=args.lr)
        # optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(start_epoch, num_epochs):
        # if epoch < len(gradual_warmup_steps):
        #     optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
        # elif epoch in lr_decay_epochs:
        #     optimizer.param_groups[0]['lr'] *= lr_decay_rate
        if epoch in args.lr_milestones:
            optimizer.param_groups[0]['lr'] /= 10.
            print(f"Lr {optimizer.param_groups[0]['lr']}")

        is_best = False
        train_metrics, val_metrics = Metrics(), Metrics()

        if not args.test:
            for i, (visual_features, boxes, question_features, answers, question_types, question_ids,
                    question_lengths) in enumerate(
                train_loader):
                visual_features = Variable(visual_features.float()).cuda()
                boxes = Variable(boxes.float()).cuda()
                question_features = Variable(question_features).cuda()
                answers = Variable(answers).cuda()

                pred = model(visual_features, boxes, question_features, answers, question_lengths)
                loss = instance_bce_with_logits(pred, answers)
                loss.backward()
                train_metrics.update_per_batch(model, answers, loss, pred, visual_features.shape[0])
                nn.utils.clip_grad_norm_(model.parameters(), 50)
                optimizer.step()
                optimizer.zero_grad()

                if i % 1000 == 0:
                    train_metrics.print(epoch)
        train_metrics.update_per_epoch()

        if None != val_loader:  # TODO: "val_loader is not None' was not working for some reason
            model.train(False)
            val_preds, val_per_type_metric = evaluate(model, val_loader, epoch, args, val_metrics)

            model.train(True)
            if val_metrics.score > best_val_score:
                best_val_score = val_metrics.score
                best_val_epoch = epoch
                is_best = True

            save_val_metrics = not args.test or args.test_has_answers
            if save_val_metrics:
                val_per_type_metric_list.append(val_per_type_metric.get_json())
                print("Best val score {} at epoch {}".format(best_val_score, best_val_epoch))
                print("Val per type scores {}".format(json.dumps(val_per_type_metric.get_json(), indent=4)))

            metrics = accumulate_metrics(epoch, train_metrics, val_metrics, val_per_type_metric,
                                         best_val_score, best_val_epoch,
                                         save_val_metrics)

            metrics_stats_list.append(metrics)

            # Add metrics + parameters of the model and optimizer
            metrics_n_model = save_metrics_n_model(metrics, model, optimizer, args, is_best)
            VqaUtils.save_stats(metrics_stats_list, val_per_type_metric_list, val_preds, args.expt_save_dir,
                                split=args.test_split, epoch=epoch)

        if args.test:
            VqaUtils.save_preds(val_preds, args.expt_save_dir, args.test_split, epoch)
            print("Test completed!")
            break


def evaluate(model, dataloader, epoch, args, val_metrics):
    per_type_metric = PerTypeMetric(epoch=epoch)
    with open(os.path.join(args.data_root, args.feature_subdir, 'answer_ix_map.json')) as f:
        answer_ix_map = json.load(f)

    all_preds = []

    for visual_features, boxes, question_features, answers, question_types, question_ids, question_lengths in iter(
            dataloader):
        visual_features = Variable(visual_features.float()).cuda()
        boxes = Variable(boxes.float()).cuda()
        question_features = Variable(question_features).cuda()

        if not args.test or args.test_has_answers:
            answers = answers.cuda()

        pred = model(visual_features, boxes, question_features, None, question_lengths)

        if not args.test or args.test_has_answers:
            loss = instance_bce_with_logits(pred, answers)
            val_metrics.update_per_batch(model, answers, loss, pred, visual_features.shape[0])

        pred_ans_ixs = pred.max(1)[1]

        # Create predictions file
        for curr_ix, pred_ans_ix in enumerate(pred_ans_ixs):
            pred_ans = answer_ix_map['ix_to_answer'][str(int(pred_ans_ix))]
            all_preds.append({
                'question_id': int(question_ids[curr_ix].data),
                'answer': str(pred_ans)
            })
            if not args.test or args.test_has_answers:
                per_type_metric.update_for_question_type(question_types[curr_ix],
                                                         answers[curr_ix].cpu().data.numpy(),
                                                         pred[curr_ix].cpu().data.numpy())
    val_metrics.update_per_epoch()
    return all_preds, per_type_metric
