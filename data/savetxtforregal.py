from scipy.io import savemat
import loaddata
import scipy
import numpy as np
import torch
import os
torch.manual_seed(0)

dataset_name = 'ppi'
for noise_level in [0, 1.0, 2.0, 3.0]:
    a1, f1, a2, f2, ground_truth = loaddata.load(dataset_name, noise_level=noise_level)
    print(f1, f2)
    feature_size = f1.size(1)
    ns = [f1.size(0), f2.size(0)]
    # edge_list_1 = get_edge_list(a1, f1.size(0))
    # edge_list_2 = get_edge_list(a2, f2.size(0))
    # features = [f1, f2]
    # edges = [a1, a2]
    f = open('ppi_combined_edges.txt', 'w')
    print(ns)
    print(a1.size())
    print(ground_truth)
    n = ns[0]
    print(n)
    t1 = (a1[0] < a1[1]).nonzero()
    t2 = (a2[0] < a2[1]).nonzero()
    a1 = a1[:, t1]
    a2 = a2[:, t2]
    g = ground_truth
    n_pre = n
    while True:
        nn1 = a1.view(-1).unique(sorted = True)
        ids = torch.zeros(n).long() -1
        ids[nn1] = torch.arange(nn1.size(0))
        
        a1 = ids[a1]
        nn2 = a2.view(-1).unique(sorted = True)
        ids2 = torch.zeros(n).long() -1
        ids2[nn2] = torch.arange(nn2.size(0)).long()
        a2 = ids2[a2]
        print(nn1.size(), nn2.size())
        gg = []
        for i in range(g.size(1)): 
            if ids[g[0, i]].item()!=-1 and ids2[g[1, i]].item()!=-1:
                gg.append([ids[g[0, i]].item(), ids2[g[1, i]].item()])
        gg = torch.tensor(gg)
        nn1 = gg[:, 0].unique()
        nn2 = gg[:, 1].unique()
        ids = torch.zeros(n).long() -1
        ids[nn1] = torch.arange(nn1.size(0))
        ids2 = torch.zeros(n).long() -1
        ids2[nn2] = torch.arange(nn2.size(0)).long()
        aa1 = []
        for i in range(a1.size(1)): 
            if ids[a1[0, i]].item()!=-1 and ids[a1[1, i]].item()!=-1:
                aa1.append([ids[a1[0, i]].item(), ids[a1[1, i]].item()])
        aa1 = torch.tensor(aa1).long()
        aa2 = []
        for i in range(a2.size(1)): 
            if ids2[a2[0, i]].item()!=-1 and ids2[a2[1, i]].item()!=-1:
                aa2.append([ids2[a2[0, i]].item(), ids2[a2[1, i]].item()])
        aa2 = torch.tensor(aa2).long()
        gg[:, 0] = ids[gg[:, 0]]
        gg[:, 1] = ids2[gg[:, 1]]
        gg = gg.t()
        aa1 = aa1.t()
        aa2 = aa2.t()
        a1 = aa1.clone().contiguous()
        a2 = aa2.clone().contiguous()
        g = gg.clone().contiguous()
        if n_pre == nn1.size(0):
            break
        else:
            n_pre = nn1.size(0)
    n = nn1.size(0)
    for i in range(aa1.size(1)):
        f.write('%d %d {\'weight\': 1.0}\n' % (aa1[0, i], aa1[1,i]))
    for i in range(aa2.size(1)):
        f.write('%d %d {\'weight\': 1.0}\n' % (aa2[0, i] + n, aa2[1,i] + n))
    print(a1[:, :10], a2[:, :10], gg[:, :10])
    # print(a1, a2)
    fk = open('ppi_edges-mapping-permutation.txt', 'wb')
    # for i in range(ground_truth.size(1)):
    np.save(fk, gg[1].numpy())
    # ss = {}
    # for i in range(ground_truth.size(1)):
    #     ss[ground_truth[0, i].item()] =  ground_truth[1, i].item()
    # import pickle
    # pickle.dump(ss, open('ppi_edges-mapping-permutation.txt', 'wb'))
    break
    # edge_lists = [edge_list_1, edge_list_2]
    # print(edge_list_1[0], edge_list_1[1])
    # print(edge_list_2[0], edge_list_2[1])
    # print(ground_truth)
    # prior = None
    # prior_rate = 0
    # g1 = scipy.sparse.csc_matrix((torch.ones(a1.size(1)).numpy(), (a1[0].numpy(), a1[1].numpy())), shape=(f1.size(0), f1.size(0)))
    # g2 = scipy.sparse.csc_matrix((torch.ones(a2.size(1)).numpy(), (a2[0].numpy(), a2[1].numpy())), shape=(f2.size(0), f2.size(0))) 
    # H = torch.zeros(f2.size(0), f1.size(0))
    # for i in range(f2.size(0)):
        # H[i] = torch.nn.functional.cosine_similarity(f1, f2[i:i+1].expand(f1.size(0), f2.size(1)), dim=-1).view(-1)
    # H = H / H.sum()
    # H =  torch.zeros(f1.size(0), f2.size(0)) + 1.0/f2.size(0)
    # print(H)
    
    # for i in g1:
    # x = { 'g1': g1.astype(np.float64), 
    # 'g1_node_label':scipy.sparse.csc_matrix(f1.numpy()).astype(np.float64),
    # 'g2': g2.astype(np.float64),
    # 'g2_node_label':scipy.sparse.csc_matrix(f2.numpy()).astype(np.float64),
    # 'g1_edge_label':np.array([[g1.astype(np.float64)]]),
    # 'g2_edge_label':np.array([[g2.astype(np.float64)]]),
    # 'H': H.numpy().astype(np.float64),
    # 'ground_truth': (ground_truth + 1).transpose(0, 1).numpy().astype(np.uint16),
    # }
    # savemat('data/final/{name}{noise}.mat'.format(name=dataset_name, noise=noise_level), x)