import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist


class GraphNet(nn.module):
	def __init__(self, image_size = 28, pred_edge = True):
		super(GraphNet, self).__init__()
		self.pred_edge = pred_edge
		N = image_size ** 2 # Number of pixels in the image
		self.fc = nn.Linear(N, 10, bias = False)
		# Create the adjacency matrix of size (N X N)
		if pred_edge:
			# Learn the adjacency matrix
			col, row = np.meshgrid(np.arange(image_size), np.arange(image_size)) # (28 x 28) Explanation: https://www.geeksforgeeks.org/numpy-meshgrid-function/
			coord = np.stack((col, row), axis = 2).reshape(-1, 2)  # (784 x 2)
			coord_normalized = (coord - np.mean(coord, axis = 0)) / (np.std(coord, axis = 0) + 1e-5) # Normalize the matrix
			coord_normalized = torch.from_numpy(coord_normalized).float() # (784 x 2)
			adjacency_matrix = torch.cat((coord_normalized.unsqueeze(0).repeat(N, 1,  1),
                                    coord_normalized.unsqueeze(1).repeat(1, N, 1)), dim=2) # (784 x 784 x 4)
			self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
				                              nn.ReLU(), 
				                              nn.Linear(64, 1),
				                              nn.Tanh())
			self.register_buffer('adjacency_matrix', adjacency_matrix)
		else:
			# Use a pre-computed adjacency matrix
			pass


	def forward(self, x):
		'''
		x: image (batch_size x 1 x image_width x image_height)
		'''
		B = x.size(0) # 64
		if self.pred_edge:
			self.A = self.pred_edge_fc(self.adjacency_matrix).squeeze() # (784 x 784) --> predicted edge map

		avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), 
											x.view(B, -1, 1)).view(B, -1)) # (64 X 784)
		return self.fc(avg_neighbor_features)



