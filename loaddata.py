import numpy as np
import os
from scipy.io import loadmat
import torch
import torch_geometric


def load_final(dataset_name):
	x = loadmat(os.path.join('data', 'final', '{name}.mat'.format(name=dataset_name)))
	if dataset_name=='douban':
		return (x['online_edge_label'][0][1], 
			x['online_node_label'], 
			x['offline_edge_label'][0][1], 
			x['offline_node_label'],
			x['ground_truth'].T,
			x['H'])

def load_arena(dataset_name, noise=0):
	n = 1135
	edges = torch.tensor(np.loadtxt('data/arena/arenas_combined_edges.txt'), dtype=torch.long).transpose(0,1).squeeze()
	row,col = edges
	edge2 = torch_geometric.utils.to_undirected(edges[:, (edges[0, :] >= n).nonzero().view(-1) ] - n).clone().squeeze()
	edge1 = torch_geometric.utils.to_undirected(edges[:, (edges[0, :] < n).nonzero().view(-1) ]).clone().squeeze()
	try:
		feature = np.load('data/arena/attr1-2vals-prob%f.npy' % noise)
		feature = torch.tensor(feature)
	except:
		feature = np.load('data/arena/attr1-2vals-prob0.000000', allow_pickle=True)
		feature = torch.tensor(feature)
		ft_sel = torch.randperm(feature.size(0))[:int(noise * feature.size(0))]
		feature[ft_sel] = 1 - feature[ft_sel]
		np.save('data/arena/attr1-2vals-prob%f' % noise, feature.numpy())
	ft_rich = torch.rand(2, 50, dtype=torch.float)
	ft_rich = ft_rich[feature.view(-1)]

	feature1 = ft_rich[:n, :]
	feature2 = ft_rich[n:, :]
	ledge = edge2.size(1)
	perm_mapping = torch.tensor(np.loadtxt('data/arena/arenas_mapping.txt'), dtype=torch.long).transpose(0, 1)
	return edge1, feature1, edge2, feature2, perm_mapping
	
def load_dbp(dataset_name, language='en_fr'):
	dataset = torch_geometric.datasets.DBP15K('../data/dbp15k', language)
	edge1 = dataset[0].edge_index1
	edge2 = dataset[0].edge_index2
	feature1 = dataset[0].x1.view(dataset[0].x1.size(0), -1)
	feature2 = dataset[0].x2.view(dataset[0].x2.size(0), -1)
	ground_truth = torch.cat( (dataset[0].train_y, dataset[0].test_y), dim=-1)
	return edge1, feature1, edge2, feature2, ground_truth
	
def load_geometric(dataset_name, noise_level = 0, noise_type = 'uniform'):
	if dataset_name == 'ppi':
		dataset = torch_geometric.datasets.PPI(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'ppi'))
	edge1 = dataset[0].edge_index
	feature1 = dataset[0].x
	edge2 = edge1.clone()
	ledge = edge2.size(1)
	edge2 = edge2[:, torch.randperm(ledge)[:int(ledge*0.9)]]
	perm = torch.randperm(feature1.size(0))
	perm_back = torch.tensor(list(range(feature1.size(0))))
	perm_mapping = torch.stack([perm_back, perm])
	edge2 = perm[edge2.view(-1)].view(2, -1) 
	edge2 = edge2[:, torch.argsort(edge2[0])]
	feature2 = torch.zeros(feature1.size())
	feature2[perm] = feature1.clone()
	if noise_type == 'uniform':
		feature2 = feature2 + 2 * (torch.rand(feature2.size())-0.5) * noise_level
	elif noise_type == 'normal':
		feature2 = feature2 + torch.randn(feature2.size()) * noise_level
	return edge1, feature1, edge2, feature2, perm_mapping
	
def load(dataset_name='cora', noise_level=0):
	if dataset_name in ['ppi']:
		return load_geometric(dataset_name, noise_level=noise_level, noise_type='uniform')
	elif dataset_name in ['douban']:
		return load_final(dataset_name)
	elif dataset_name in ['arena']:
		return load_arena(dataset_name, noise=noise_level)