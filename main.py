import time
start = time.perf_counter()
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
import numpy as np
import os
import random
import pickle
from tqdm import tqdm
from scipy.io import loadmat
from typing import List
from sklearn.metrics import confusion_matrix
from GraphVAE import VAE_LL_loss
from utilis import Calculate_TC


criterion = nn.CrossEntropyLoss()

def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=50,
                        help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument("--emb_normlize", type = bool, default=  False, help="mlp hidden dims")
parser.add_argument("--data_split_ratio", type=float, default= [0.8, 0.1, 0.1], help="data seperation")
parser.add_argument("--latent_dim", type=int, default=  [128,128], help="classifier hidden dims")
parser.add_argument('--readout', type=str, default="mean", choices=["sum", "average", "max"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument("--mlp_hidden", type=int, default=  [64,64], help="mlp hidden dims")
parser.add_argument("--GVAE_hidden_dim", type = int, default= 64, help="mlp hidden dims")
parser.add_argument("--mi_weight", type=float, default= 0.001, help="classifier hidden dims")
parser.add_argument("--weight_decay", type=float, default=0.001, help="Adam weight decay. Default is 5*10^-5.")
parser.add_argument("--input_dim", type=int, default=110)
parser.add_argument("--output_dim", type=int, default=3)
parser.add_argument("--lambda1", type=float, default=0.001)
parser.add_argument("--lambda2", type=float, default=0.0001)
parser.add_argument("--lambda3", type=float, default=0.001)
parser.add_argument("--num_prototypes", type=int, default=2)
args = parser.parse_args()

#set up seeds and gpu device
torch.manual_seed(0)
np.random.seed(0)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --- load data ---
from prepare_data import get_dataloader,load_dataset
graph_filename = "data/mm_dataset.pkl"
pyg_dataset = pickle.load(open(graph_filename, "rb"))
#dataset = load_dataset(pyg_dataset)
dataloader = get_dataloader(pyg_dataset, args.batch_size, data_split_ratio=args.data_split_ratio, seed=args.seed)

# --- train/load GCE ---
from GraphVAE import GraphEncoder,GraphDecoder
encoder = GraphEncoder(args.input_dim, args.GVAE_hidden_dim, device).to(device)
decoder = GraphDecoder(args.GVAE_hidden_dim, args.input_dim).to(device)

# --- train/load subgraph ---
from pre_subgraph import Prot_subgraph
from GIN_classifier import GINNet
from sklearn.model_selection import KFold
model = GINNet(args.input_dim, args.output_dim, args, device).to(device)
SG_model = Prot_subgraph(device, encoder, decoder, model, args.GVAE_hidden_dim, args.num_prototypes, args.dropout).to(device)


def train_0(args, model, device, train_graphs, SG_model,opt):
    model.eval()
    SG_model.eval()
    encoder.train()
    decoder.train()
    for graph in train_graphs: 
        x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
        z, mu, logvar= encoder(graph)
        Xhat, adj = decoder(z)
        # #print(Xhat.shape)
        nll = VAE_LL_loss(graph.x, Xhat, logvar, mu, device)
        graph_prot = torch.zeros(args.num_prototypes, len(z[:,0]),round(args.GVAE_hidden_dim /args.num_prototypes)).to(device)
        for k in range(args.num_prototypes):
            prot_size_up = round(args.GVAE_hidden_dim * k/args.num_prototypes)
            prot_size_de = round(args.GVAE_hidden_dim * (k+1)/args.num_prototypes)
            edge = z[:,prot_size_up:prot_size_de]
            #print(edge.shape)
            graph_prot[k,:,:]= edge
        TC_loss = Calculate_TC(graph_prot, args.num_prototypes)
        prototype_loss = nll + 0.01 * TC_loss
        #print(prototype_loss)
        if opt is not None:
            opt.zero_grad()
            prototype_loss.backward()    
            opt.step()
        
        return graph_prot

def train_20(args, model, device, train_graphs, SG_model,opt,optimizer, num_prototype, log_file, epoch):
        model.train()
        SG_model.train()
        encoder.train()
        decoder.train()

        acc_accum = 0
        num = 0
        labels_train = []
        cluster_assignments_train = []
        prototype_edges = []
        prototype_activations = []
        ids = []

        for graph in train_graphs:
            ids.append(graph.id) 
            x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
            logits, loss, prototype_edge, prototype_activation, cluster_assignment = SG_model(graph,args.lambda2 * 2)
            
            prototype_activations.append(prototype_activation.detach().cpu().numpy())
            prototype_edges.append(prototype_edge.detach().cpu().numpy())
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()      
                optimizer.step()

            loss = loss.detach().cpu().numpy()
            pred = logits.max(1, keepdim=True)[1]
            labels = graph.y.clone().detach().to(device)
            labels_train.append(labels.cpu().numpy())
            cluster_assignments_train.append(cluster_assignment.cpu().numpy())
            #print(pred.shape)
            #print(labels.shape)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(graph.y))
            acc_accum = acc_accum + acc
            num = num + 1
                    
        acc_train = acc_accum/num
        print("classification loss: %f" %(loss))
        print("accuracy train: %f" %(acc_train))

        return logits, loss, acc_train, prototype_edges, prototype_activations, cluster_assignments_train, labels_train, ids

def evaluate(args, data, model, SG_model, device):
    model.eval()
    SG_model.eval()

    acc_accum = 0
    num = 0
    all_cluster_assignments = []
    all_labels = []
    prototype_edges = []
    ids = []
    for graph in data:
        ids.append(graph.id)
        x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
        logits, loss, prototype_edge, prototype_activations, cluster_assignment= SG_model(graph, args.lambda2 * 2)
        prototype_edges.append(prototype_edge.detach().cpu().numpy())
        all_cluster_assignments.append(cluster_assignment.cpu().numpy())
        pred = logits.max(1, keepdim=True)[1]
        labels = graph.y.clone().detach().to(device)
        all_labels.append(labels.cpu().numpy())
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc = correct / float(len(graph.y))
        acc_accum += acc
        num += 1

    return acc_accum / num, np.concatenate(all_cluster_assignments), np.concatenate(all_labels), ids, prototype_edges

def calc_performance_statistics(y_pred, y):
    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    num_classes = cm.shape[0]  # 类别数
    
    # 初始化全局统计量
    TN_total = 0
    FP_total = 0
    FN_total = 0
    TP_total = 0
    
    # 对每个类别计算 TP, FP, FN, TN
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        TN_total += TN
        FP_total += FP
        FN_total += FN
        TP_total += TP
    
    # 计算全局指标
    N = TN_total + TP_total + FN_total + FP_total
    S = (TP_total + FN_total) / N
    P = (TP_total + FP_total) / N
    acc = (TN_total + TP_total) / N
    sen = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0
    spc = TN_total / (TN_total + FP_total) if (TN_total + FP_total) != 0 else 0
    prc = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0
    f1s = 2 * (prc * sen) / (prc + sen) if (prc + sen) != 0 else 0
    mcc = (TP_total / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P)) if (P * S * (1 - S) * (1 - P)) != 0 else 0

    return acc, sen, spc, prc, f1s, mcc

def test(args, data, model, SG_model, device): 

    model.eval()
    SG_model.eval()

    acc_accum = 0
    sen_accum = 0
    spc_accum = 0
    prc_accum = 0
    f1s_accum = 0
    mcc_accum = 0
    num = 0
    all_cluster_assignments = []
    all_labels = []
    prototype_edges = []
    ids = []
    for graph in data:
        ids.append(graph.id) 
        x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
        logits, loss, prototype_edge, prototype_activations, cluster_assignment = SG_model(graph, args.lambda2 * 2)
        prototype_edges.append(prototype_edge.detach().cpu().numpy())
        #compute gradient
        pred = logits.max(1, keepdim=True)[1]
        labels = graph.y.clone().detach().to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
         # 保存结果到列表
        all_cluster_assignments.append(cluster_assignment.cpu().numpy())
        all_labels.append(labels)
        if len(labels)>1:
            test_acc, test_sen, test_spc, test_prc, test_f1s, test_mcc = calc_performance_statistics(pred,labels)
            acc = correct / float(len(graph.y))
            acc_accum = acc_accum + acc
            sen_accum = sen_accum + test_sen
            spc_accum = spc_accum + test_spc
            prc_accum = prc_accum + test_prc
            f1s_accum = f1s_accum + test_f1s
            mcc_accum = mcc_accum + test_mcc
            num = num + 1
        acc_test = acc_accum/num
        sen_test = sen_accum/num
        spc_test = spc_accum/num
        prc_test = prc_accum/num
        f1s_test = f1s_accum/num
        mcc_test = mcc_accum/num
        print(f"test accuracy: {acc_test}")

    return acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test, \
    np.concatenate(all_cluster_assignments), np.concatenate(all_labels), ids, prototype_edges

kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
fold_idx = 0

for train_index, test_index in kf.split(pyg_dataset):
    train_dataset = [pyg_dataset[i] for i in train_index]
    test_dataset = [pyg_dataset[i] for i in test_index]
    eval_index = int(len(test_dataset) * 0.5)
    eval_dataset = test_dataset[:eval_index]
    test_dataset = test_dataset[eval_index:]

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        'eval': DataLoader(eval_dataset, batch_size = len(eval_dataset), shuffle=False),
        'test': DataLoader(test_dataset, batch_size = len(test_dataset), shuffle=False)
    }

    encoder = GraphEncoder(args.input_dim, args.GVAE_hidden_dim, device).to(device)
    decoder = GraphDecoder(args.GVAE_hidden_dim, args.input_dim).to(device)
    model = GINNet(args.input_dim, args.output_dim, args, device).to(device)
    SG_model = Prot_subgraph(device, encoder, decoder, model, args.GVAE_hidden_dim, args.num_prototypes, args.dropout).to(device)

    opt_params = list(decoder.parameters()) + list(encoder.parameters())
    opt = torch.optim.Adam(opt_params, lr=args.lr , weight_decay=args.weight_decay)
    params = list(model.parameters()) + list(SG_model.parameters())
    optimizer = torch.optim.Adam( params, lr =args.lr , weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
    log_file = f"context/trainLog_{fold_idx}.txt"

    # best_eval_acc = 0
    # patience = 20
    # no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print('Epoch: {}'.format(epoch))

        if epoch >=20:
            logits, avg_loss, acc_train, prototype_edges, prototype_activations, cluster_assignments_train, labels_train, ids = \
            train_20(args, model, device,  dataloader['train'], SG_model,opt,optimizer, args.num_prototypes, log_file, epoch)

            acc_eval, cluster_assignments_eval, labels_eval, ids_eval, prototype_edges_eval = \
            evaluate(args, dataloader['eval'], model, SG_model, device)

            print("accuracy eval: %f" %(acc_eval))
            with open(log_file, 'a+') as f:
                f.write("Epoch %d, loss: %f, acc_train: %f, acc_eval: %f\n" % (epoch, avg_loss, acc_train, acc_eval))
        else:
            graph_prot = train_0(args, model, device,  dataloader['train'], SG_model, opt)
        
        if epoch == args.epochs:  # 仅保存最后一个 epoch 的结果
            np.savez(f'context/train_fold_{fold_idx}.npz',
                assignments=cluster_assignments_train,
                edges=prototype_edges,
                activations=prototype_activations,
                labels=labels_train,
                ids=ids)
            np.savez(f'context/eval_fold_{fold_idx}.npz',
                assignments=cluster_assignments_eval,
                labels=labels_eval,
                edges=prototype_edges_eval,
                ids=ids_eval)
        scheduler.step()
    
    acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test, cluster_assignments_test, labels_test, \
    ids_test, prototype_edges_test = test(args, dataloader['test'], model, SG_model, device)

    np.savez(f'context/test_fold_{fold_idx}.npz',
        assignments=cluster_assignments_test,
        labels=labels_test,
        edges=prototype_edges_test,
        ids=ids_test)
    
    print("accuracy test: %f" %(acc_test))
    with open(log_file, 'a+') as f:
        f.write("acc_test: %f, sen_test: %f, spc_test: %f, prc_test: %f, f1s_test: %f, mcc_test: %f \n\n" \
                % (acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test))
    fold_idx += 1

end = time.perf_counter()
print(f"Time elapsed: {end - start:.6f} seconds")