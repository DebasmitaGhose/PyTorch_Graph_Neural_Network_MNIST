# PyTorch_Graph_Neural_Network_MNIST

Example code to train a Graph Neural Network on the MNIST dataset in PyTorch for Digit Classification

## References

- Detailed Explanation: https://medium.com/@BorisAKnyazev/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-1-3d9fada3b80d
- Implementation: https://github.com/bknyaz/examples/blob/master/fc_vs_graph_train.py

## Running the Code

To use precomputed adjacency matrix
`python gnn_mnist.py`

To use a learned edge map
`python gnn_mnist.py --pred_edge`

Other optional hyperparameters:
`python gnn_mnist.py --pred_edge --batch_size 64 --epochs 10 --lr 1e-4 --seed 10`


