import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist


class GraphNet(nn.Module):
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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Cross entropy loss
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description = 'GNN PyTorch Example on MNIST')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--pred_edge', action='store_true', default=True, 
                        help='predict edges instead of using predefined ones')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed')
    args = parser.parse_args()

    use_cuda = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)    
    
    model = GraphNet()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = 1e-4)

    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
    


