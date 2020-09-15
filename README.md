# Wasserstein Distance Graph Alignment

Run the code:

```
python main.py --dataset (dataset_name) --setup (sid) [--prior_rate 0.02 --use_config (True/False) ...]
```

--datasets ppi/douban/arena
Currently supports three datasets: PPI, Douban, Arena Emails

--setup 1/2/3/4
1 - GCN+Pseudo
2 - GAT+Pseudo
3 - LGCN+Pseudo
4 - LGCN+Wasserstein

--use_config True/False
if use_config is True (Default to be True), the config file in folder configs will be loaded.

--prior_rate 
Only for Douban data, by default use 2% of the prior matrix.