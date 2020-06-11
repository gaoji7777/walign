from scipy.io import savemat
import loaddata
import scipy
import numpy as np
import torch
import os
from tqdm import tqdm
dataset_name = 'ppi'
for noise_level in [0, 1.0, 2.0, 3.0]:
    a1, f1, a2, f2, ground_truth = loaddata.load(dataset_name, noise_level=noise_level)
    print(f1, f2)
    feature_size = f1.size(1)
    ns = [f1.size(0), f2.size(0)]
    edges = [a1, a2]
    print(ground_truth)
    prior = None
    prior_rate = 0
    g1 = scipy.sparse.csc_matrix((torch.ones(a1.size(1)).numpy(), (a1[0].numpy(), a1[1].numpy())), shape=(f1.size(0), f1.size(0)))
    g2 = scipy.sparse.csc_matrix((torch.ones(a2.size(1)).numpy(), (a2[0].numpy(), a2[1].numpy())), shape=(f2.size(0), f2.size(0))) 
    H = torch.zeros(f2.size(0), f1.size(0))
    batch = 1024
    for i in tqdm(range(f2.size(0))):
        for j in range(f1.size(0) // batch + 1):
            f1now = f1[j * batch:(j+1)* batch, :]
            H[i, j * batch:(j+1)* batch] = torch.nn.functional.cosine_similarity(f1now, f2[i:i+1].expand(f1now.size(0), f2.size(1)), dim=-1).view(-1)
    H = H / H.sum()
    print(H)
    x = { 'g1': g1.astype(np.float64), 
    'g2': g2.astype(np.float64),
    'g1_edge_label':np.array([[g1.astype(np.float64)]]),
    'g2_edge_label':np.array([[g2.astype(np.float64)]]),
    'H': H.numpy().astype(np.float64),
    'ground_truth': (ground_truth + 1).transpose(0, 1).numpy().astype(np.uint16),
    }
    savemat('data/final/{name}{noise}.mat'.format(name=dataset_name, noise=noise_level), x)