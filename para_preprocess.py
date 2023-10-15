import os
import pickle
import networkx as nx
import multiprocessing
import concurrent.futures
import pickle
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import argparse
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='gat')
parser.add_argument('--layout', type=str, default='fdl') #kk
parser.add_argument('--dis', type=str, default='dis') #rdf
parser.add_argument('--num_layout', type=int, default=8)
parser.add_argument('--rwse_wl', type=int, default=20)


args, unknown = parser.parse_known_args()
print(args)
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

start_time = time.perf_counter()

def calculate_distances(layout_seed, layout_type, G, edge_index, dim):
    set_seed(layout_seed)
    if layout_type == 'fdl':
        pos = nx.spring_layout(G, seed=layout_seed, dim=dim)
        edge_distances = torch.zeros(edge_index.shape[1])
        for j in range(edge_index.shape[1]):
            u = edge_index[0, j].item()
            v = edge_index[1, j].item()
            u_pos = torch.tensor(pos[u])
            v_pos = torch.tensor(pos[v])
            distance = torch.norm(u_pos - v_pos)
            edge_distances[j] = distance.item()
        return edge_distances

    if layout_type == 'kk':
        pos = nx.kamada_kawai_layout(G, dim=dim)
        edge_distances = torch.zeros(edge_index.shape[1])
        for j in range(edge_index.shape[1]):
            u = edge_index[0, j].item()
            v = edge_index[1, j].item()
            u_pos = torch.tensor(pos[u])
            v_pos = torch.tensor(pos[v])
            distance = torch.norm(u_pos - v_pos)
            edge_distances[j] = distance.item()
        return edge_distances

if os.path.exists(feature_path):
    with open(feature_path, 'rb') as f:
        edge_attrs = pickle.load(f)
        print('Start training !!!')
else:
    import time
    args = {'layout': 'kk', 'dim': 2} 
    start_time = time.perf_counter()

    results = []
    batch_size = 20

    # batch process
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        with multiprocessing.Pool(processes=16) as pool:  # 
            results = []

            for data in batch:
                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                G.add_edges_from(data.edge_index.t().tolist())
                edge_index = data.edge_index

                layout_seeds = range(num_layouts)
                layout_types = [args['layout']] * num_layouts
                dims = [args['dim']] * num_layouts

                result = pool.starmap(
                    calculate_distances,
                    zip(layout_seeds, layout_types, [G] * num_layouts, [edge_index] * num_layouts, dims)
                )
                results.append(result)

        #  Now, the results contains the edges under each layout_ Distances 
        for result in results:
            edge_attrs.append(torch.stack(result, dim=1).float())

    #  Process remaining data points (if any) 
    if len(dataset) % batch_size != 0:
        remaining_data = dataset[len(dataset) - (len(dataset) % batch_size):]
        with multiprocessing.Pool(processes=8) as pool: 
            results = []

            for data in remaining_data:
                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                G.add_edges_from(data.edge_index.t().tolist())
                edge_index = data.edge_index

                layout_seeds = range(num_layouts)
                layout_types = [args['layout']] * num_layouts
                dims = [args['dim']] * num_layouts

                result = pool.starmap(
                    calculate_distances,
                    zip(layout_seeds, layout_types, [G] * num_layouts, [edge_index] * num_layouts, dims)
                )
                results.append(result)
        for result in results:
            edge_attrs.append(torch.stack(result, dim=1).float())

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("graph layout runing times", execution_time, "s")
    with open(feature_path, 'wb') as f:
        pickle.dump(edge_attrs, f)
