import torch
import torch.nn.functional as F
import sklearn.neighbors
from loss import pairwise_loss, feature_reconstruct_loss

def pred_anchor_links_from_embd(trans, networks, mode='cosine', prior=None, prior_rate=0):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))
	
	cossim = torch.zeros(embd1.size(0), embd0.size(0))
	for i in range(embd1.size(0)):
		cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
	if prior is not None:
		cossim = (1 + cossim)/2 * (1-prior_rate) + prior * prior_rate
	ind = cossim.argmax(dim=1)

	anchor_links = torch.zeros(ind.size(0), 2, dtype=torch.long)
	anchor_links[:, 0] = ind.view(-1)
	anchor_links[:, 1] = torch.arange(ind.size(0))
	return anchor_links

def train_adv(trans, optimizer_trans, discriminator, optimizer_d, networks, batch_d_per_iter=10, batch_size_align=512):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))
	trans.train()
	discriminator.train()
	models[0].train()
	models[1].train()
	embd0_copy = embd0.clone().detach()
	embd1_copy = embd1.clone().detach()	

	for j in range(batch_d_per_iter):
		optimizer_d.zero_grad()
		loss = F.binary_cross_entropy(discriminator(embd0_copy), torch.full((embd0_copy.size(0), 1), 0.0, dtype=torch.float)) + \
			F.binary_cross_entropy(discriminator(embd1_copy), torch.full((embd1_copy.size(0), 1), 1.0, dtype=torch.float))
		loss.backward()
		optimizer_d.step()

	return F.binary_cross_entropy(discriminator(embd1), torch.full((embd1.size(0), 1), 0.0, dtype=torch.float))

def train_supervise_align(anchor_links, trans, optimizer_trans, networks, margin=0.2, batch_size_align=128, mode='cosine'):
	models = [t[0] for t in networks]
	optimizers = [t[1] for t in networks]
	features = [t[2] for t in networks]
	ns = [t.size(0) for t in features]
	edges = [t[3] for t in networks]
	avg_loss = .0
	neg_sampling_matrix = get_neg_sampling_align(anchor_links, ns[0])
	for j in range(anchor_links.size(0)//batch_size_align + 1):
		neg_losses = []
		embd0 = models[0](features[0], edges[0])
		embd1 = models[1](features[1], edges[1])
		embd1_trans = trans(embd1[anchor_links[j * batch_size_align: j * batch_size_align + batch_size_align, 1], :])
		embd0_curr = embd0[anchor_links[j * batch_size_align: j * batch_size_align + batch_size_align, 0], :]
		if mode=='cosine':
			dist = 1 - F.cosine_similarity(embd1_trans, embd0_curr)
		else:
			dist = F.mse_loss(embd1_trans, embd0_curr, reduction='none').sum(dim=1)
		for ik in range(2, neg_sampling_matrix.size(1)):
			if mode=='cosine':
				neg_losses.append(1 - F.cosine_similarity(embd1_trans, embd0[neg_sampling_matrix[j * batch_size_align: j * batch_size_align + batch_size_align, ik], :]))
			else:
				neg_losses.append(F.mse_loss(embd1_trans, embd0[neg_sampling_matrix[j * batch_size_align: j * batch_size_align + batch_size_align, ik], :], reduction='none').sum(dim=1))
		neg_losses = torch.stack(neg_losses, dim=-1)
		loss_p = torch.max(margin - dist.unsqueeze(-1) + neg_losses, torch.FloatTensor([0])).sum()
		avg_loss += loss_p
	avg_loss /= anchor_links.size(0)
	return avg_loss
	
def get_neg_sampling_align(anchor_links, n, neg_sampling_num=5):
	neg_sampling_matrix = torch.zeros(anchor_links.size(0), neg_sampling_num + 2, dtype=torch.long)
	neg_sampling_matrix[:, 0] = anchor_links[:, 0]
	neg_sampling_matrix[:, 1] = anchor_links[:, 1]
	for i in range(anchor_links.size(0)):
		ss = torch.ones(n, dtype=torch.long)
		ss[anchor_links[i, 0]] = 0
		nz = torch.nonzero(ss).view(-1)
		neg_sampling_matrix[i, 2:] = nz[torch.randperm(nz.size(0))[:neg_sampling_num]]
	return neg_sampling_matrix

def test(model, feature, train_edges, test_edges, edge_list):
	model.eval()
	with torch.no_grad():
		x = model(feature, train_edges.transpose(0, 1))
		neg_sampling_matrix = get_neg_sampling(test_edges, edge_list)
		loss = calculate_loss(x, test_edges, neg_sampling_matrix)
		print(loss.item()/test_edges.size(0))

def train_prior(trans, optimizer_trans, networks, prior):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))
	
	cossim = torch.zeros(embd1.size(0), embd0.size(0))
	for i in range(embd1.size(0)):
		cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
	
	return (cossim-prior).norm()
	
def train_uniform(trans, optimizer_trans, networks, prior):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))
	
	cossim = torch.zeros(embd1.size(0), embd0.size(0))
	for i in range(embd1.size(0)):
		cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
	
	return cossim.norm()
	
def train_wgan_adv_pseudo_self( trans, optimizer_trans, wdiscriminator, optimizer_d, networks, lambda_gp=10, batch_d_per_iter=5, batch_size_align=512):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))

	trans.train()
	wdiscriminator.train()
	models[0].train()
	models[1].train()

	for j in range(batch_d_per_iter):
		w0 = wdiscriminator(embd0)
		w1 = wdiscriminator(embd1)
		anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
		anchor0 = w0.view(-1).argsort(descending=False)[: embd1.size(0)]
		embd0_anchor = embd0[anchor0, :].clone().detach()
		embd1_anchor = embd1[anchor1, :].clone().detach()
		optimizer_d.zero_grad()
		loss = -torch.mean(wdiscriminator(embd0_anchor)) + torch.mean(wdiscriminator(embd1_anchor))
		loss.backward()
		optimizer_d.step()
		for p in wdiscriminator.parameters():
			p.data.clamp_(-0.1, 0.1)
	w0 = wdiscriminator(embd0)
	w1 = wdiscriminator(embd1)
	anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
	anchor0 = w0.view(-1).argsort(descending=False)[: embd1.size(0)]
	embd0_anchor = embd0[anchor0, :]
	embd1_anchor = embd1[anchor1, :]
	loss = -torch.mean(wdiscriminator(embd1_anchor))
	return loss


def train_feature_recon(trans, optimizer_trans, networks, recon_models, optimizer_recons, batch_r_per_iter=10):
	models = [t[0] for t in networks]
	features = [t[2] for t in networks]
	edges = [t[3] for t in networks]
	recon_model0, recon_model1 = recon_models
	optimizer_recon0, optimizer_recon1 = optimizer_recons
	embd0 = models[0](features[0], edges[0])
	embd1 = trans(models[1](features[1], edges[1]))
	
	recon_model0.train()
	recon_model1.train()
	trans.train()
	models[0].train()
	models[1].train()
	embd0_copy = embd0.clone().detach()
	embd1_copy = embd1.clone().detach()	
	for t in range(batch_r_per_iter):
		optimizer_recon0.zero_grad()
		loss = feature_reconstruct_loss(embd0_copy, features[0], recon_model0)
		loss.backward()
		optimizer_recon0.step()
	for t in range(batch_r_per_iter):
		optimizer_recon1.zero_grad()
		loss = feature_reconstruct_loss(embd1_copy, features[1], recon_model1)
		loss.backward()
		optimizer_recon1.step()
	loss = 0.5 * feature_reconstruct_loss(embd0, features[0], recon_model0) + 0.5 * feature_reconstruct_loss(embd1, features[1], recon_model1)

	return loss
	
		
def check_align(embds, ground_truth, k=5, mode='cosine', prior=None, prior_rate=0):
	embd0, embd1 = embds
	g_map = {}
	for i in range(ground_truth.size(1)):
		g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
	g_list = list(g_map.keys())
	
	cossim = torch.zeros(embd1.size(0), embd0.size(0))
	for i in range(embd1.size(0)):
		cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
	if prior is not None:
		cossim = (1 + cossim)/2 * (1-prior_rate) + prior * prior_rate
	
	ind = cossim.argsort(dim=1, descending=True)[:, :k]
	a1 = 0
	ak = 0 
	for i, node in enumerate(g_list):
		if ind[node, 0].item() == g_map[node]:
			a1 += 1
			ak += 1
		else:
			for j in range(1, ind.shape[1]):
				if ind[node, j].item() == g_map[node]:
					ak += 1
					break
	a1 /= len(g_list)
	ak /= len(g_list)
	print('H@1 %.2f%% H@5 %.2f%%' % (a1*100, ak*100))
	return a1, ak
	
def check_align_greedy(embds, ground_truth, k=5, mode='cosine', prior=None, prior_rate=0):
	embd0, embd1 = embds
	g_map = {}
	for i in range(ground_truth.size(1)):
		g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
	g_list = list(g_map.keys())
	
	cossim = torch.zeros(embd1.size(0), embd0.size(0))
	for i in range(embd1.size(0)):
		cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)

	if prior is not None:
		cossim = (1 + cossim)/2 * (1-prior_rate) + prior * prior_rate
	ind = cossim.argsort(dim=1, descending=True)[:, :k]
	cossim_flatten_ind = cossim.view(-1).argsort(descending=True)
	cc = 0
	row = torch.zeros(embd1.size(0), dtype=bool)
	col = torch.zeros(embd0.size(0), dtype=bool)
	for i in range(embd1.size(0)):
		if row[cossim_flatten_ind[cc] / embd0.size(0)].item() or col[cossim_flatten_ind[cc] % embd0.size(0)].item():
			cc += 1
		else:
			nrow = cossim_flatten_ind[cc] / embd0.size(0)
			ncol = cossim_flatten_ind[cc] % embd0.size(0)
			oldcol = -1
			for s in range(k):
				if ind[nrow, s] == ncol:
					oldcol = s
					break
			if oldcol!=0:
				if oldcol==-1:
					ind[nrow, 1:] = ind[nrow, :k-1]
					ind[nrow, 0] = ncol
				else:
					ind[nrow, 1: oldcol+1] = ind[nrow, : oldcol]
					ind[nrow, 0] = ncol
			
			row[nrow] = True
			col[ncol] = True
			cc += 1
	a1 = 0
	ak = 0 
	for i, node in enumerate(g_list):
		if ind[node, 0].item() == g_map[node]:
			a1 += 1
			ak += 1
		else:
			for j in range(1, ind.shape[1]):
				if ind[node, j].item() == g_map[node]:
					ak += 1
					break
	a1 /= len(g_list)
	ak /= len(g_list)
	print('H@1 %.2f%% H@5 %.2f%%' % (a1*100, ak*100))
	return a1, ak
	
