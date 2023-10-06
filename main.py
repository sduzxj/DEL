import torch
import os
import pickle
import torch.nn as nn
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

import torch.nn.functional as F
from torch_geometric.nn import (
    MLP,
    GINConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GATv2Conv,
    GATConv,
    TransformerConv,
    GINEConv,
    GCNConv,
    GPSConv,
)
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from utils import *

# 加载TUDataset
import argparse
from torch_geometric.logging import init_wandb, log
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--edge_dim', type=int, default=16)
parser.add_argument('--use_edge', type=str, default='force')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--update', type=str, default='t')
parser.add_argument('--model', type=str, default='gat')
parser.add_argument('--layout', type=str, default='fdl') #kk
parser.add_argument('--dis', type=str, default='dis') #rdf
parser.add_argument('--num_layout', type=int, default=8) #kk
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
edge_attrs=[]
data_list=[]
num_layouts=args.num_layout

feature_path=''
if args.layout =='fdl':
     feature_path='./layout_feature/'+args.dataset+'_features.pkl'
if args.layout =='kk':
     feature_path='./layout_feature/'+args.dataset+'_kk_features.pkl'
if args.layout =='sfdl':
    feature_path='./layout_feature/'+args.dataset+'_sfdl_features.pkl'

if os.path.exists(feature_path):
    with open(feature_path, 'rb') as f:
        edge_attrs = pickle.load(f)
        print('Start training !!!')
else:
    import time
    start_time = time.perf_counter()
    for data in dataset:
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.t().tolist())
        edge_distances = torch.zeros(data.edge_index.shape[1], num_layouts)
        #########Calculate the distance matrix using the Force-Directed Layout algorithm.
        for i in range(num_layouts):
            if args.layout=='fdl':
                pos = spring_layout(G, seed=i, dim=args.dim, sampling='base')
                for j in range(data.edge_index.shape[1]):
                    u = data.edge_index[0, j].item()
                    v = data.edge_index[1, j].item()
                    u_pos = torch.tensor(pos[u])
                    v_pos = torch.tensor(pos[v])
                    distance = torch.norm(u_pos - v_pos)
                    edge_distances[j, i] = distance.item()
        #########Calculate the distance matrix using the Kamada-Kawai Layout layout algorithm.
            if args.layout=='kk':
                set_seed(i)
                pos = kamada_kawai_layout(G, dim=args.dim)
                for j in range(data.edge_index.shape[1]):
                    u = data.edge_index[0, j].item()
                    v = data.edge_index[1, j].item()
                    u_pos = torch.tensor(pos[u])
                    v_pos = torch.tensor(pos[v])
                    distance = torch.norm(u_pos - v_pos)
                    edge_distances[j, i] = distance.item()
            edge_attr = edge_distances.float()
        edge_attrs.append(edge_attr)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("graph layout runing times", execution_time, "s")
    with open(feature_path, 'wb') as f:
        pickle.dump(edge_attrs, f)

# index=0

for index, data in enumerate(dataset):
    edge_attr = None 

    if args.dis == 'rdf':
        dis = torch.pow(edge_attrs[index].reshape(-1, num_layouts), 2)
        dis = torch.exp(-dis)

    elif args.dis == 'dis':
        dis = edge_attrs[index].reshape(-1, num_layouts)

    elif args.dis == 'random':
        dis = torch.rand(edge_attrs[index].size(0), num_layouts)

    if args.model == 'GPS':
        data = Data(x=data.x, edge_index=data.edge_index, edge_attr=dis, y=data.y, pe=data.pe)

    else:
        data = Data(x=data.x, edge_index=data.edge_index, edge_attr=dis, y=data.y)

    data_list.append(data)

# for data in dataset:
#             if args.dis=='rdf':
#                dis=torch.pow(edge_attrs[index].reshape(-1, num_layouts), 2)
#                dis =  torch.exp(-dis)
#             if args.dis=='dis':
#                 dis=edge_attrs[index].reshape(-1, num_layouts)
#             if args.dis=='random':
#                 dis=torch.rand(edge_attrs[index].size(0), num_layouts)
#             if args.model=='GPS':
#                  data = Data(x=data.x, edge_index=data.edge_index, edge_attr=dis, y=data.y,pe=data.pe)
#             else: 
#                  data = Data(x=data.x, edge_index=data.edge_index, edge_attr=dis, y=data.y)            
#             data_list.append(data)
#             index=index+1
##########

class PermutationInvariantNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PermutationInvariantNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc_stats = nn.Linear(output_dim+3, output_dim)  # 3表示要计算的统计量数量

    def forward(self, x):
        # Sort x
        x_sorted, _ = torch.sort(x, dim=1)

        # Calculate original features
        x_features = self.fc1(x_sorted)
        x_features = F.relu(x_features)
        x_features = self.fc2(x_features)

        # Calculate statistics for each sample
        x_range = torch.max(x_sorted, dim=1)[0] - torch.min(x_sorted, dim=1)[0]  # Range
        x_std = torch.std(x_sorted, dim=1)  # Standard deviation
        x_max = torch.max(x_sorted, dim=1)[0]  # Maximum value

        # Combine original features and statistics
        x_combined = torch.cat((x_features, x_range.unsqueeze(1), x_std.unsqueeze(1), x_max.unsqueeze(1)), dim=1)

        # Process the combined features through another layer if needed
        x_combined = F.relu(self.fc_stats(x_combined))
        return x_combined

# 构建GNN模型
class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes,use_edge):
        super(GNN, self).__init__()
        hidden_channels=args.hidden_channels
        self.PM=PermutationInvariantNet(num_layouts, hidden_channels ,edge_dims)
        self.lin1 = nn.Linear(hidden_channels, num_classes) 
        self.lin2 = nn.Linear(2*hidden_channels, edge_dims) 
        self.lin3 = nn.Linear(2*edge_dims, edge_dims) 

        if args.model=='gat':

            self.conv1 = GATv2Conv(num_features, hidden_channels, edge_dim=edge_dims)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels,edge_dim=edge_dims)

        if args.model=='GT':

            self.conv1 = TransformerConv(num_features, hidden_channels,edge_dim =edge_dims )
            self.conv2 = TransformerConv(hidden_channels, hidden_channels,edge_dim =edge_dims) 

        if args.model=='GPS':

            self.PM=PermutationInvariantNet(num_layouts, hidden_channels ,hidden_channels)
            self.mlp0 = MLP([num_features, hidden_channels, hidden_channels])
            mlp1 = MLP([hidden_channels, hidden_channels, hidden_channels])
            mlp2 = MLP([hidden_channels, hidden_channels, hidden_channels])
            self.lin2 = nn.Linear(2*hidden_channels, hidden_channels) 
            self.lin3 = nn.Linear(2*hidden_channels, hidden_channels) 
            self.conv1 =  GPSConv(hidden_channels, GINEConv(mlp1,edge_dim=hidden_channels), heads=4)
            self.conv2 = GPSConv(hidden_channels, GINEConv(mlp2,edge_dim=hidden_channels), heads=4)

        self.use_edge= use_edge
        
    def forward(self, data, batch):
        if args.model=='GPS':
                x, edge_index, edge_attr, pe = data.x, data.edge_index, data.edge_attr, data.pe
        else:
                 x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if args.model=='GPS':
              x=self.mlp0(torch.cat((x,pe),dim=1))
              edge_attr= self.PM(edge_attr)
              x = self.conv1(x, edge_index,batch,edge_attr=edge_attr)
              edge_attr = self.update_edge_embedding(edge_attr, x[edge_index[0]], x[edge_index[1]])
              x = F.relu(x)
              x = self.conv2(x, edge_index,batch,edge_attr=edge_attr)
              x = F.relu(x)
        else: 
              edge_attr= self.PM(edge_attr)
              x = self.conv1(x, edge_index,edge_attr=edge_attr)
              edge_attr = self.update_edge_embedding(edge_attr, x[edge_index[0]], x[edge_index[1]])
              x = F.relu(x)
              x = self.conv2(x, edge_index,edge_attr=edge_attr)
              x = F.relu(x)
        x  = self.lin1(global_add_pool(x , batch))
        return F.log_softmax(x, dim=1)
    
    def update_edge_embedding(self, edge_attr, x_i, x_j):
        message=self.lin2 (torch.cat((x_i , x_j),dim=1))
        updated_edge_attr = self.lin3(torch.cat((edge_attr,message),dim=1))  
        return updated_edge_attr


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

# test model
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
np.random.shuffle(data_list)
dataset = Batch.from_data_list(data_list)

for fold in range(k):
   fold_size = len(dataset) // k
   start_idx = fold * fold_size
   end_idx = (fold + 1) * fold_size
   patience=25
   max_patience=25
   best_val_acc=0
   best_test_acc=0
   best_train_acc=0
   
   val_dataset = dataset[start_idx:end_idx]
   train_dataset = dataset[:start_idx] + dataset[end_idx:]
   train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
   val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = GNN(num_features, num_classes, use_edge=args.use_edge).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
         
        if val_acc > best_val_acc:
                 best_val_acc=val_acc
                 best_train_acc=train_acc
                 patience = max_patience
        else :
               patience =patience-1
        if patience ==0:
               break
   best_val_acc_list.append(best_val_acc)
   best_train_acc_list.append(best_train_acc)

print('####')
print('mean',np.mean(best_val_acc_list))
print('std',np.std(best_val_acc_list))
print('####')

filename = './result/'+args.dataset+'_'+args.dis+args.layout+args.model+args.use_edge+'_'+'_ed'+str(args.edge_dim)+'_d'+str(args.hidden_channels)+'.txt'
with open(filename, 'a') as file:
    file.write(f"Seed: {args.seed}\n")
    np.savetxt(file, [np.mean(best_val_acc_list)], fmt='%.6f')
