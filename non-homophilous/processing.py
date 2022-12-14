import numpy as np
import torch
import pickle
import argparse
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from dataset import load_nc_dataset

### Parse args ###
parser = argparse.ArgumentParser(description='Processing Pipeline')
parser.add_argument('--data_name', type=str, default='wiki', help='datasets.') #wiki pokec
parser.add_argument('--K', type=int, default=10, help='propagation steps.')
args = parser.parse_args()


def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#Loading dataset
dataset = load_nc_dataset(args.data_name)
print("Loading completed!")

edge_index = dataset.graph['edge_index']
N = dataset.graph['num_nodes']
print("Node:", N, "Edge:", edge_index.shape)

print('Making the graph undirected')
edge_index = to_undirected(edge_index)

#Load edges and create adjacency
row,col = edge_index
row=row.numpy()
col=col.numpy()
adj_mat=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))


print("Get scaled laplacian matrix.")
adj_mat = sys_normalized_adjacency(adj_mat)
adj_mat = -1.0*adj_mat #\hat{L}=2/(L*lambda_max)-I, lambda_max=2, \hat{L}=L-I=-P
adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)

cheb_embedding = []

#T_0(\hat{L})X
T_0_feat = dataset.graph['node_feat']
cheb_embedding.append(T_0_feat)

# del for free
del dataset, edge_index, row, col

#T_1(\hat{L})X
T_1_feat = torch.spmm(adj_mat,T_0_feat)
cheb_embedding.append(T_1_feat)

# compute T_k(\hat{L})X
print("Begining of iteration!")
for i in range(1,args.K):
    #T_k(\hat{L})X
    T_2_feat = torch.spmm(adj_mat,T_1_feat)
    T_2_feat = 2*T_2_feat-T_0_feat
    T_0_feat, T_1_feat =T_1_feat, T_2_feat
    cheb_embedding.append(T_2_feat)
    print("Done:",i)

torch.save(cheb_embedding, './data/cheb_'+args.data_name+'.pt')
print(args.data_name+" has been successfully processed")

