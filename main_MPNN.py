import torch
import os
import pickle
import torch.nn as nn
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

import torch.nn.functional as F
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_mean_pool, global_max_pool, GATv2Conv,TransformerConv, GINEConv, GCNConv, GPSConv, MLP
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


import argparse
from torch_geometric.logging import init_wandb, log
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--edge_dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='gat')
parser.add_argument('--rwse_wl', type=int, default=20) #kk


args, unknown = parser.parse_known_args()
print(args)

import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
set_seed(args.seed)
# Add rwse embedding
rwse_wl=args.rwse_wl
if args.model=='GPS':
    transform = T.AddRandomWalkPE(walk_length=rwse_wl, attr_name='pe')


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
    


if 'IMDB' in args.dataset or 'COLLAB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                if 'COLLAB' in args.dataset :
                      return data.num_nodes <= 125
                else: 
                      return data.num_nodes <= 70
                      
        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                if 'COLLAB' in args.dataset : 
                       data.x = F.one_hot(data.x, num_classes=125).to(torch.float)
                else:
                       data.x = F.one_hot(data.x, num_classes=69).to(torch.float)
                return data
        if args.model=='GPS':
               transform=T.Compose([MyPreTransform(),transform])
               dataset = TUDataset(
            root='./data/RWSE',
            name=args.dataset,
            pre_transform=transform,
            pre_filter=MyFilter())
        else :
               transform=MyPreTransform()
               dataset = TUDataset(
            root='./data',
            name=args.dataset,
            pre_transform=transform,
            pre_filter=MyFilter())
else:
         if args.model=='GPS':
              dataset = TUDataset(root='./data/RWSE', name=args.dataset,pre_transform=transform)
         else:
              dataset = TUDataset(root='./data', name=args.dataset)#, use_node_attr=True)

num_features=dataset.num_features
if args.model=='GPS':
      num_features=num_features+20

num_classes=dataset.num_classes
edge_dims=args.edge_dim
new_dataset = []
data_list=[]

for index, data in enumerate(dataset):
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
    data_list.append(data)


class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        hidden_channels=args.hidden_channels
        self.lin1 = nn.Linear(hidden_channels, num_classes) 
        self.lin2 = nn.Linear(2*num_features, edge_dims) 
        self.lin3 = nn.Linear(2*hidden_channels, edge_dims) 
        if args.model=='gat':
            self.conv1 = GATv2Conv(num_features, hidden_channels, edge_dim=edge_dims)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels,edge_dim=edge_dims)
        if args.model=='gine':
            self.lin2 = nn.Linear(2*num_features, hidden_channels) 
            self.lin3 = nn.Linear(2*hidden_channels, hidden_channels) 
            mlp1 = MLP([num_features, hidden_channels, hidden_channels])
            mlp2 = MLP([hidden_channels, hidden_channels, hidden_channels])
            self.conv1 = GINEConv(nn=mlp1, edge_dim=hidden_channels)
            self.conv2 = GINEConv(nn=mlp2,edge_dim=hidden_channels) 
        if args.model=='GT':
            self.conv1 = TransformerConv(num_features, hidden_channels,edge_dim =edge_dims )
            self.conv2 = TransformerConv(hidden_channels, hidden_channels,edge_dim =edge_dims) 
        if args.model=='GPS':
            self.mlp0 = MLP([num_features, hidden_channels, hidden_channels])
            mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels])
            mlp2 = MLP([hidden_channels, hidden_channels, hidden_channels])
            self.lin2 = nn.Linear(2*hidden_channels, hidden_channels) 
            self.lin3 = nn.Linear(2*hidden_channels, hidden_channels) 
            self.conv1 =  GPSConv(hidden_channels, GINEConv(mlp1,edge_dim=hidden_channels), heads=4)
            self.conv2 = GPSConv(hidden_channels, GINEConv(mlp2,edge_dim=hidden_channels), heads=4)
        
    def forward(self, data, batch):
        if args.model=='GPS':
                x, edge_index, pe = data.x, data.edge_index, data.pe
                x=self.mlp0(torch.cat((x,pe),dim=1))
        else:
                 x, edge_index = data.x, data.edge_index

        edge_attr= self.lin2 (torch.cat((x[edge_index[0]], x[edge_index[1]]),dim=1))
        x = self.conv1(x, edge_index,edge_attr=edge_attr)
        edge_attr= self.lin3 (torch.cat((x[edge_index[0]], x[edge_index[1]]),dim=1))#self.update_edge_embedding(x[edge_index[0]], x[edge_index[1]])
        x = F.relu(x)
        x = self.conv2(x, edge_index,edge_attr=edge_attr)
        x = F.relu(x)
        x  = self.lin1(global_add_pool(x , batch))
        return F.log_softmax(x, dim=1)

def train(epoch):
    total_loss =0
    model.train()
    for epoch in range(10):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data, data.batch).argmax(dim=1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

best_val_acc_list=[]
best_train_acc_list=[]
k=10
#set_seed(42)  # 设置随机种子，以确保结果可重现
#random.seed(42)

np.random.shuffle(data_list)
dataset = Batch.from_data_list(data_list)

for fold in range(k):
   fold_size = len(dataset) // k
   start_idx = fold * fold_size
   end_idx = (fold + 1) * fold_size
   patience=20
   best_val_acc=0
   best_test_acc=0
   best_train_acc=0
   
   val_dataset = dataset[start_idx:end_idx]
   train_dataset = dataset[:start_idx] + dataset[end_idx:]
   train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
   val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   #set_seed(42)
   model = GNN(num_features, num_classes).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        #test_acc = test(test_loader)
         
        if val_acc > best_val_acc:
                 best_val_acc=val_acc
                 #best_test_acc=test_acc
                 best_train_acc=train_acc
                 patience =25
        else :
               patience =patience
        if patience ==0:
               break
        #log(Epoch=epoch, Loss=loss, Train=train_acc, Test=best_val_acc)   
   print(best_val_acc)
   best_val_acc_list.append(best_val_acc)
   best_train_acc_list.append(best_train_acc)
#Note: We use 5 different seeds to calculate the actual standard deviation; 
#the printed standard deviation is based solely on cross-fold validation.
print('####')
print(np.mean(best_val_acc_list))
print(np.std(best_val_acc_list))
print('####')

filename = './result/'+args.dataset+'_MPNN_'+args.model+'_'+'_ed'+str(args.edge_dim)+'_d'+str(args.hidden_channels)+'.txt'
with open(filename, 'a') as file:
    file.write(f"Seed: {args.seed}\n")
    np.savetxt(file, [np.mean(best_val_acc_list)], fmt='%.6f')
