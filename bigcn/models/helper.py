import os

import numpy as np
import io

import torch


class Evaluator:
    def __init__(self, args):
        self.args = args

        self.saved_model = None
        self.curr_epoch = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.val_acc = 0
        self.test_acc = 0

        self.batch_values = None
        self.batch_count = 0

    def add_loss(self, *values):
        if isinstance(values[0], torch.Tensor):
            values = [v.item() for v in values]
        values = np.array(values)
        if self.batch_values is None:
            self.batch_values = values
        else:
            self.batch_values += values
        self.batch_count += 1

    def flush_loss(self):
        out = self.batch_values / self.batch_count
        self.batch_values = None
        self.batch_count = 0
        fmt_list = ['{:.3f}'] * out.shape[0]
        return '\t'.join(fmt.format(e) for fmt, e in zip(fmt_list, out))

    def check_epoch(self, epoch, loss, model):
        self.curr_epoch = epoch
        if loss < self.best_loss:
            self.best_epoch = epoch
            self.best_loss = loss
            self.saved_model = io.BytesIO()
            torch.save(model.state_dict(), self.saved_model)
            return True
        return False

    def is_stop(self):
        return self.curr_epoch >= self.best_epoch + self.args.patience

    def restore_best_model(self, model):
        self.saved_model.seek(0)
        model.load_state_dict(torch.load(self.saved_model))

    def record_accuracy(self, val_acc, test_acc):
        self.val_acc = val_acc
        self.test_acc = test_acc

    def write_log(self):
        args = self.args
        out_path = '{}/summary.tsv'.format(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'a') as f:
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                args.graph, args.seed, self.best_epoch, self.val_acc, self.test_acc))
