from typing import Any, Optional

import numpy as np
import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset
    
class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization='sym'):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        pe = torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1])
        if pe.shape[1] < self.pos_enc_dim:
            zero_padding = torch.zeros(pe.shape[0], self.pos_enc_dim - pe.shape[1])
            pe = torch.cat((pe, zero_padding), dim=1)
        return  pe.float()#torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        dataset.pe = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            g['pe']=pe
        return dataset