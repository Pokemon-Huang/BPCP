#PyTorch ResNet Cifar Pruning
seed = 1
use_gpu = False

dataset = "cifar10"
if dataset == "cifar10":
    num_classes = 10
    data_dir = '/home/yangke_huang/data/cifar10'
elif dataset == "cifar100":
    num_classes = 100
    data_dir = '/home/yangke_huang/data/cifar100'

# network
network = "ResNet"
depth = 56

# train
batch_size = 128
test_batch_size = 1

lr = 0.1
momentum = 0.9
weight_decay = 0.0001
start_epoch = 0
epoch = 160
log_interval = 100

block_pruning = False
gamma = 0.05

channel_pruning = False
sparsity = 0.0001