import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from graphconv import Unweighted, CombUnweighted

def glorot(tensor):
	if tensor is not None:
		stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
		tensor.data.uniform_(-stdv, stdv)
		
class LGCN(torch.nn.Module):
	def __init__(self, input_size, output_size, hidden_size=512, K=8):
		super(LGCN, self).__init__()
		self.conv1 = CombUnweighted(K=K)
		self.linear = torch.nn.Linear(input_size * (K + 1), output_size)
	def forward(self, feature, edge_index):
		x = self.conv1(feature, edge_index)
		x = self.linear(x)
		return x

class GATNet(torch.nn.Module):
	def __init__(self, input_size, output_size, hidden_size=512, heads=1):
		super(GATNet, self).__init__()
		self.conv1 = GATConv(input_size, hidden_size, heads=heads)
		self.conv2 = GATConv(hidden_size * heads, output_size)
	def reset_parameters(self):
		self.conv1.reset_parameters()
		self.conv2.reset_parameters()
	def forward(self, feature, edge_index):
		x = F.dropout(feature, p=0.5, training=self.training)
		x = F.elu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.conv2(x, edge_index)
		return x

class GCNNet(torch.nn.Module):
	def __init__(self, input_size, output_size, hidden_size=512):
		super(GCNNet, self).__init__()
		self.conv1 = GCNConv(input_size, hidden_size)
		self.conv2 = GCNConv(hidden_size, output_size)
	def reset_parameters(self):
		self.conv1.reset_parameters()
		self.conv2.reset_parameters()
	def forward(self, feature, edge_index):
		x = F.dropout(feature, p=0.5, training=self.training)
		x = F.elu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.conv2(x, edge_index)
		return x
	
class WDiscriminator(torch.nn.Module):
	def __init__(self, hidden_size, hidden_size2=512):
		super(WDiscriminator, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, 1)
	def forward(self, input_embd):
		return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

class transformation(torch.nn.Module):
	def __init__(self, hidden_size=512, hidden_size2=512):
		super(transformation, self).__init__()
		self.trans = torch.nn.Parameter(torch.eye(hidden_size))
	def forward(self, input_embd):
		return input_embd.mm(self.trans)
	
class notrans(torch.nn.Module):
	def __init__(self):
		super(notrans, self).__init__()
	def forward(self, input_embd):
		return input_embd

class ReconDNN(torch.nn.Module):
	def __init__(self, hidden_size, feature_size, hidden_size2=512):
		super(ReconDNN, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, feature_size)
	def forward(self, input_embd):
		return self.output(F.relu(self.hidden(input_embd)))