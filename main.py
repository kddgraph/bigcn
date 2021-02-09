import argparse
import os

import torch
import numpy as np
from torch import optim, nn

from bigcn import data, models, utils


class BiGCN(nn.Module):
    def __init__(self, args, graph, edges, num_nodes, num_features, num_classes, trn_nodes,
                 trn_features, use_laplacian=True, generator='network'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.use_laplacian = use_laplacian
        self.generator = generator

        self.ce_loss = nn.CrossEntropyLoss()
        self.lap_loss = None
        self.mse_loss = None

        if generator == 'network':
            self.gen_model = models.GCN(num_classes, args.gen_units, num_features, args.gen_dropout)
            self.mse_loss = nn.MSELoss()
        else:
            self.gen_model = models.Generator(graph, edges, num_nodes, trn_nodes, trn_features, generator)
            num_features = self.gen_model.num_features()

        if use_laplacian:
            self.lap_loss = models.LaplacianLoss()

        self.cls_model = models.GCN(num_features, args.cls_units, num_classes, args.cls_dropout)

    def features(self, beliefs, edges):
        if self.generator == 'network':
            return self.gen_model(beliefs, edges)
        else:
            return self.gen_model()

    def classify(self, features, edges):
        return self.cls_model(features, edges)

    def gcn_emb(self, features, edges):
        return self.cls_model(features, edges, emb=True)[0]

    def forward(self, x):
        pass

    def compute_losses(self, edges, trn_nodes, trn_x, trn_labels, beliefs):
        num_features = trn_x.size(1)
        pred_features = self.features(beliefs, edges)
        pred_labels = self.classify(pred_features, edges)
        losses = [self.ce_loss(pred_labels[trn_nodes], trn_labels)]
        if self.use_laplacian:
            losses.append(self.lap_loss(pred_features, edges))
        if self.generator == 'network':
            losses.append(self.mse_loss(pred_features[trn_nodes], trn_x) * num_features)
        return losses


def parse_args():
    parser = argparse.ArgumentParser()

    # Modules for ablation studies.
    parser.add_argument('--model', type=str, default='full')

    # for experiments
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--graph', type=str, default='cora')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--print-every', type=int, default=0)
    parser.add_argument('--no-save', action='store_true', default=False)
    parser.add_argument('--obs-ratio', type=float, default=0.5)

    # hyperparameters
    parser.add_argument('--cls-units', type=int, default=32)
    parser.add_argument('--cls-dropout', type=float, default=0.)
    parser.add_argument('--gen-units', type=int, default=32)
    parser.add_argument('--gen-dropout', type=float, default=0.)
    parser.add_argument('--epsilon', type=float, default=2.)

    return parser.parse_args()


def evaluate(model, indices, labels, idx, beliefs):
    model.eval()
    pred_features = model.features(beliefs, indices)
    pred_labels = model.classify(pred_features, indices)
    return utils.accuracy(pred_labels[idx], labels[idx]).item()


def main():
    args = parse_args()
    args.out = os.path.join(args.out, args.model)
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = args.graph
    device = utils.to_device(args.gpu)

    edges, features, labels, trn_nodes, val_nodes, test_nodes = data.load_data(graph, args.obs_ratio)
    num_nodes = features.size(0)
    num_features = features.size(1)
    num_classes = labels.max().item() + 1

    trn_x = features[trn_nodes]
    trn_y = labels[trn_nodes]
    beliefs = models.BPModel(edges, num_nodes, num_classes, trn_nodes, trn_y, args.epsilon).to(device)()

    if args.model == 'lbp':
        val_acc = utils.accuracy(beliefs[val_nodes], labels[val_nodes]).item()
        test_acc = utils.accuracy(beliefs[test_nodes], labels[test_nodes]).item()
        print('{}\t{}\t{}\t{}'.format(args.graph, args.seed, val_acc, test_acc))
        return
    elif args.model in ['param', 'onehot', 'node2vec', 'svd']:
        use_laplacian = False
        generator = args.model
    elif args.model == 'laplacian':
        use_laplacian = True
        generator = 'param'
    elif args.model == 'full':
        use_laplacian = True
        generator = 'network'
    else:
        raise ValueError(args.model)
    model = BiGCN(args, graph, edges, num_nodes, num_features, num_classes, trn_nodes, trn_x,
                  use_laplacian, generator).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    log_path = os.path.join(args.out, 'logs/{}-{}.tsv'.format(graph, seed))
    model_path = os.path.join(args.out, 'models/{}-{}.pth'.format(graph, seed))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(log_path):
        os.remove(log_path)

    edges = edges.to(device)
    labels = labels.to(device)
    trn_nodes = trn_nodes.to(device)
    trn_x = trn_x.to(device)
    trn_y = trn_y.to(device)

    if args.print_every > 0:
        print('epoch\tl_ce\tl_lap\tl_mse\tl_all\tis_best')

    evaluator = models.Evaluator(args)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = model.compute_losses(edges, trn_nodes, trn_x, trn_y, beliefs)
        evaluator.add_loss(*losses)

        optimizer.zero_grad()
        torch.stack(losses).sum().backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            trn_losses = model.compute_losses(edges, trn_nodes, trn_x, trn_y, beliefs)
            trn_loss = torch.stack(trn_losses).sum()

        with open(log_path, 'a') as f:
            log = '{:5d}\t{}\t{:.3f}'.format(epoch, evaluator.flush_loss(), trn_loss)
            if evaluator.check_epoch(epoch, trn_loss, model):
                log += '\tBEST'
            f.write(log + '\n')

            if args.print_every > 0 and epoch % args.print_every == 0:
                print(log)

        if evaluator.is_stop():
            break

    evaluator.restore_best_model(model)
    if not args.no_save:
        torch.save(model.state_dict(), model_path)
    val_acc = evaluate(model, edges, labels, val_nodes, beliefs)
    test_acc = evaluate(model, edges, labels, test_nodes, beliefs)

    log = '{}\t{}\t{}\t{}'.format(args.graph, args.seed, val_acc, test_acc)
    print(log)
    if args.print_every > 0:
        print()
    with open(os.path.join(args.out, 'summary.txt'), 'a') as f:
        f.write(log + '\n')


if __name__ == '__main__':
    main()
