import torch
from tqdm import tqdm

def get_adj_list(edge_index, n=None):
	# Convert edge list into adjacency list
	if not n:
		n = torch.max(edge_index)
	edge_map = {it:None for it in range(n)}
	ccd = -1
	pre_pos = 0
	for edge_id in tqdm(range(len(edge_index[0]))):
		if ccd != edge_index[0, edge_id].item():
			if ccd != -1:
				edge_map[ccd] = edge_index[1, pre_pos:edge_id]
			pre_pos = edge_id
			ccd = edge_index[0, edge_id].item()
	edge_map[ccd] = edge_index[1, pre_pos:]
	for node in edge_map: # self loop
		if edge_map[node] is not None:
			edge_map[node] = torch.cat((torch.LongTensor([node]), edge_map[node]), dim=0)
		else:
			edge_map[node] = torch.LongTensor([node])
	adj_list = []
	for node in range(n):
		adj_list.append(edge_map[node])
	return adj_list