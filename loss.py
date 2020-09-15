import torch
import torch.nn.functional as F

def pairwise_loss(x1ns, y1ns):
	# f(x_1, y_1, x_1_neighbors, y_1_neighbors)
	# x_1 has anchor link with y1
	# x_2 is neighbor with x_1, y_2 is the neighbor of y_1
	# Push up maximum similarity between pairs of x1ns and y1ns
	l1 = x1ns.size(0)
	l2 = y1ns.size(0)
	x1ns = x1ns.unsqueeze(1).expand(-1, l2, -1)
	y1ns = y1ns.unsqueeze(0).expand(l1, -1, -1)
	cos_sim = F.cosine_similarity(x1ns, y1ns, dim=-1)
	loss = - cos_sim.mean()
	return loss
	
def feature_reconstruct_loss(embd, x, recon_model):
	recon_x = recon_model(embd)
	return torch.norm(recon_x - x, dim=1, p=2).mean()