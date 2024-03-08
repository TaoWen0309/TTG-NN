import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset

from tqdm import trange

from util import separate_TUDataset, compute_PI_tensor
from models.tensorgcn import TenGCN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, train_PIs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = trange(total_iters, unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        batch_PI = torch.stack([train_PIs[idx] for idx in selected_idx])
        output = model(batch_graph,batch_PI)

        labels = torch.LongTensor([graph.y for graph in batch_graph]).to(device)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

def pass_data_iteratively(model, graphs, PIs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        output.append(model(batch_graph,batch_PI).detach())
    return torch.cat(output, 0)

def test(model, device, test_graphs, test_PIs):
    model.eval()

    output = pass_data_iteratively(model, test_graphs, test_PIs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    
    print("accuracy test: %f" % acc_test)
    return acc_test

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--tensor_layer_type', type = str, default = "TCL", choices=["TCL","TRL"],
                                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--node_pooling', action="store_false",
    					help='node pooling based on node scores')
    # NOTE
    # PROTEINS: ['degree','betweenness','closeness']
    # ENZYMES: ['degree','betweenness','eigenvector','closeness']
    # DD: ['degree','betweenness','eigenvector','closeness']
    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','eigenvector','closeness'],
    					help='Methods for sublevel filtration on PDs')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')
    args = parser.parse_args()

    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    graphs = TUDataset(root='/tmp/' + args.dataset, name=args.dataset)
    num_classes = graphs.num_classes

    ## NOTE: compute graph PI tensor if necessary
    # PIs = compute_PI_tensor(graphs,args.PI_dim,args.sublevel_filtration_methods)
    # torch.save(PIs,'{}_{}_PI.pt'.format(args.dataset,args.PI_dim))
    ## load pre-computed PIs
    PIs = torch.load('{}_{}_PI.pt'.format(args.dataset,args.PI_dim)).to(device)
    print('finished loading PI for dataset {} with PI_dim = {}'.format(args.dataset,args.PI_dim))
    
    train_graphs, train_PIs, test_graphs, test_PIs = separate_TUDataset(graphs, PIs, args.seed, args.fold_idx)
    model = TenGCN(args.num_layers, args.num_mlp_layers, train_graphs[0].x.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.tensor_layer_type, args.node_pooling, args.PI_dim, args.sublevel_filtration_methods, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print("Current epoch is:", epoch)

        avg_loss = train(args, model, device, train_graphs, train_PIs, optimizer, epoch)
        scheduler.step()
        acc_test = test(model, device, test_graphs, test_PIs)

        max_acc = max(max_acc, acc_test)

        if not args.filename == "":
            with open(args.filename, 'a+') as f:
                f.write("%f %f %f" % (avg_loss, acc_test))
                f.write("\n")

    with open('acc_results.txt', 'a+') as f:
        f.write(str(args.dataset) + ' ' + str(args.fold_idx) + ' ' + str(max_acc) + '\n')

if __name__ == '__main__':
    main()