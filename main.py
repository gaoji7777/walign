from loaddata import load, load_final
import numpy as np
import torch
import itertools
seed = 1
torch.manual_seed(seed)
import torch.nn.functional as F
import torch_geometric
from graphmodel import WDiscriminator, LGCN, GCNNet, GATNet, transformation, ReconDNN, notrans
from train import train_wgan_adv_pseudo_self, check_align, train_supervise_align, pred_anchor_links_from_embd, train_feature_recon
from preprocess import get_adj_list
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=int, default=4)
parser.add_argument('--dataset', type=str, default='ppi')
parser.add_argument('--use_config', type=bool, default=True)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--transformer', type=int, default=1)
parser.add_argument('--prior_rate', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_wd', type=float, default=0.01)
parser.add_argument('--lr_recon', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

if args.use_config:
	try:
		import json
		f = open('configs/%s.config' % args.dataset, 'r')
		arg_dict = json.load(f)
		for t in arg_dict:
			args.__dict__[t] = arg_dict[t]
	except:
		print('Error in loading config and use default setting instead')

print(args)
if args.setup==1:
	args.net = GCNNet
elif args.setup==2:
	args.net = GATNet
elif args.setup==3 or args.setup==4:
	args.net = LGCN
	
dataset_name=args.dataset
noise_level = args.noise
if dataset_name in ['douban']:
	a1, f1, a2, f2, ground_truth, prior = load(dataset_name, noise_level=noise_level)
	feature_size = f1.shape[1]
	ns = [a1.shape[0], a2.shape[0]]
	edge_1 = torch.LongTensor(np.array(a1.nonzero()))
	edge_2 = torch.LongTensor(np.array(a2.nonzero())) 
	ground_truth = torch.tensor(np.array(ground_truth, dtype=int)) - 1  # Original index start from 1
	features = [torch.FloatTensor(f1.todense()), torch.FloatTensor(f2.todense())]
	edges = [edge_1, edge_2]
	prior = torch.FloatTensor(prior)
	prior_rate = args.prior_rate
elif dataset_name in ['ppi', 'arena']:
	a1, f1, a2, f2, ground_truth = load(dataset_name, noise_level=noise_level)
	feature_size = f1.size(1)
	ns = [f1.size(0), f2.size(0)]
	features = [f1, f2]
	edges = [a1, a2]
	prior = None
	prior_rate = 0
	
	
mode='cosine'

check_align(features, ground_truth, mode=mode, prior=prior, prior_rate=prior_rate)

num_graph = 2
networks = []
feature_output_size = args.hidden_size

torch.seed()

model = args.net(feature_size, args.hidden_size)
optimizer = None
for i in range(num_graph):
	networks.append((model, optimizer, features[i], edges[i]))
if args.transformer==1 and args.setup!=2 and args.setup!=3:
	trans = transformation(args.hidden_size)
else:
	trans = notrans()
optimizer_trans = torch.optim.Adam(itertools.chain(trans.parameters(), networks[0][0].parameters()), lr=args.lr, weight_decay=5e-4)

embd0 = networks[0][0](features[0], edges[0])
embd1 = networks[1][0](features[1], edges[1])
with torch.no_grad():
	a1, ak = check_align([embd0, trans(embd1)], ground_truth, mode=mode, prior=prior, prior_rate=prior_rate)

wdiscriminator = WDiscriminator(feature_output_size)
optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.lr_wd, weight_decay=5e-4)

recon_model0 = ReconDNN(feature_output_size, feature_size)
recon_model1 = ReconDNN(feature_output_size, feature_size)
optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=args.lr_recon, weight_decay=5e-4)
optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=args.lr_recon, weight_decay=5e-4)

batch_size_align = 128

best = 0
bp = 0,0

time1 = time.time()
for i in range(1, args.epochs + 1):
	trans.train()
	networks[0][0].train()
	networks[1][0].train()
	
	optimizer_trans.zero_grad()
	if args.setup==1 or args.setup==2 or args.setup==3:
		anchor_links_pred = pred_anchor_links_from_embd(trans, networks, mode=mode, prior=prior, prior_rate=prior_rate)
		loss = train_supervise_align(anchor_links_pred, trans, optimizer_trans, networks, mode=mode)
	elif args.setup==4:
		loss = train_wgan_adv_pseudo_self(trans, optimizer_trans, wdiscriminator, optimizer_wd, networks)

	loss_feature = train_feature_recon(trans, optimizer_trans, networks, [recon_model0, recon_model1], [optimizer_recon0, optimizer_recon1])
	loss = (1-args.alpha) * loss + args.alpha * loss_feature
	
	loss.backward()
	optimizer_trans.step()
	
	networks[0][0].eval()
	networks[1][0].eval()
	trans.eval()
	embd0 = networks[0][0](features[0], edges[0])
	embd1 = networks[1][0](features[1], edges[1])

	
	with torch.no_grad():
		a1, ak = check_align([embd0, trans(embd1)], ground_truth, mode=mode, prior=prior, prior_rate=prior_rate)
	if a1 > best:
		best = a1
		bp = a1, ak
		
time2 = time.time()
print('Total Time %.2f' % (time2-time1))
print('H@1 %.2f%% H@5 %.2f%%' % (bp[0]*100, bp[1]*100))
f_rec = open('record.txt', 'a+')
f_rec.write('Data %s Setup %d Noise %.2f H@1 %.2f%% H@5 %.2f%% Time %.2f\n' % (args.dataset, args.setup, args.noise, bp[0]*100, bp[1]*100, time2-time1))
