import torch.nn as nn
import time
import os
import json


class Metrics:
    """
    Stores accuracy (score), loss and timing info
    """

    def __init__(self):
        self.loss = 0
        self.raw_score = 0
        self.score = 0
        self.normalized_score = 0
        self.start_time = time.time()
        self.end_time = 0
        self.total_norm = 0
        self.count_norm = 0
        self.num_examples = 0
        self.upper_bound = 0
        self.reset_start_time()

    def update_per_batch(self, model, answers, loss, pred, curr_size):
        upper_bound = answers.max(1)[0].sum()
        self.upper_bound += upper_bound
        # self.total_norm += nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # self.count_norm += 1
        from train import compute_score_with_logits
        batch_score = compute_score_with_logits(pred, answers.data).sum()
        self.loss += loss.data * curr_size
        self.raw_score += batch_score
        self.num_examples += curr_size

    def update_per_epoch(self):
        self.loss /= self.num_examples
        self.raw_score = self.raw_score / self.num_examples
        self.upper_bound = self.upper_bound / self.num_examples
        self.normalized_score = self.raw_score / self.upper_bound
        self.score = self.raw_score
        self.end_time = time.time()

    def print(self, epoch):
        print("Epoch {} Score {:.2f} Loss {}".format(epoch, 100 * self.raw_score / self.num_examples,
                                                      self.loss / self.num_examples))

    def reset_start_time(self):
        self.start_time = time.time()


def accumulate_metrics(epoch, train_metrics, val_metrics, val_per_type_metric,
                       best_val_score,
                       best_val_epoch, save_val_metrics=True):
    stats = {
        "epoch": epoch,

        "train_loss": float(train_metrics.loss),
        "train_raw_score": float(train_metrics.raw_score),
        "train_normalized_score": float(train_metrics.normalized_score),
        "train_upper_bound": float(train_metrics.upper_bound),
        "train_score": float(train_metrics.score),
        "train_num_examples": train_metrics.num_examples,

        "train_time": train_metrics.end_time - train_metrics.start_time,
        "val_time": val_metrics.end_time - val_metrics.start_time
    }
    if save_val_metrics:
        stats["val_raw_score"] = float(val_metrics.raw_score)
        stats["val_normalized_score"] = float(val_metrics.normalized_score)
        stats["val_upper_bound"] = float(val_metrics.upper_bound)
        stats["val_loss"] = float(val_metrics.loss)
        stats["val_score"] = float(val_metrics.score)
        stats["val_num_examples"] = val_metrics.num_examples
        stats["val_per_type_metric"] = val_per_type_metric.get_json()

        stats["best_val_score"] = float(best_val_score)
        stats["best_epoch"] = best_val_epoch

    print(json.dumps(stats, indent=4))
    return stats
