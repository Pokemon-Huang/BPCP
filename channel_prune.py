from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import time
import shutil
import argparse
import numpy as np
from collections import OrderedDict
from models import *
import config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default=0.2,
                        help='scale sparse rate (default: 0.2)')
    parser.add_argument('--save_path', default='./exp/channel_prune', type=str, metavar='PATH',
                        help='path to save dir')
    parser.add_argument('--prune', default='./exp/finetune/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to pretrained model')
    return parser.parse_args()

# initial
block_pruning = True
channel_pruning = True
args = get_args()
use_cuda = config.use_gpu and torch.cuda.is_available()
torch.manual_seed(config.seed)
if use_cuda:
    torch.cuda.manual_seed(config.seed)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# dataset
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
if config.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(config.data_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(config.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(config.data_dir, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(config.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)

# model
checkpoint = torch.load(args.prune)
model = ResNet_v2_channel(depth=config.depth, block_pruning=block_pruning,
                          block_cfg=checkpoint['block_cfg'], num_classes=config.num_classes)
model.load_state_dict(checkpoint['state_dict'])
if use_cuda:
    model.cuda()
#print(model)

# prune
# print(model.gate)
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
#print(total)

#for layer,(name,module) in enumerate(model.named_modules()):
#    print(layer)
#    print(name)
#    print(module)

modules = list(model.named_modules())
def get_pt():
    prune_table = [[] for _ in range(len(modules))]
    not_prune = []
    curser = 0
    layer = 0
    while layer < len(modules):
        name, module = modules[layer]
        if name.startswith('layer') and len(name)<7:
            if model.gate[curser] == 0:
                cap = layer + 2
                layer = cap + 1
            else:
                cap = layer + 10
                layer += 1
            curser += 6
            continue
        if isinstance(module, Bottleneck_channal):
            prune_table[cap].append(layer + 6)
            not_prune.append(layer + 6)
            layer += 8
        else: layer += 1
    return prune_table, not_prune
pruneTable,notPrune = get_pt()
print(pruneTable)
print(notPrune)

def get_key(list, value):
    for k in range(len(list)):
        if value in list[k]:
            return k
    return None

class Pruner:
    def __init__(self, model, num_pruned):
        self.model = model
        self.num_filters_to_prune = num_pruned
        self.scale_dict = OrderedDict()
        self.importance_dict = OrderedDict()

    def get_pruning_plan(self):
        self.compute_bn()
        self.get_importance()
        return self.get_pruning(self.num_filters_to_prune)

    def compute_bn(self):
        module_list = list(self.model.modules())
        for index, module in enumerate(module_list):
            if isinstance(module, nn.Conv2d):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                scale = module.weight.data.abs().cpu().numpy()
                self.scale_dict[layer] = scale

    def get_importance(self):
        #print(self.scale_dict.keys())
        for layer, scale in self.scale_dict.items():
            #print("layer", layer)
            #print("scale", scale.shape)

            if layer in notPrune:
                l = get_key(pruneTable, layer)
                if l not in self.importance_dict.keys():
                    self.importance_dict[l] = scale
                else:
                    self.importance_dict[l] = np.maximum(self.importance_dict[l], scale)
            else:
                self.importance_dict[layer] = scale

    def get_pruning(self, num_filters_to_prune):
        print(self.importance_dict.keys())
        #for layer, importance in self.importance_dict.items():
         #   print("layer", layer)
          #  print("importance", importance)
        filters_to_prune_per_layer = {}
        i = 0
        while i < num_filters_to_prune:
            argmin_within_layers = list(map(np.argmin, list(self.importance_dict.values())))
            min_within_layers = list(map(np.min, list(self.importance_dict.values())))
            argmin_cross_layers = np.argmin(np.array(min_within_layers))
            cut_layer_name = list(self.importance_dict.keys())[int(argmin_cross_layers)]
            cut_layer_index = list(self.importance_dict.keys())[int(argmin_cross_layers)]
            cut_channel_index = argmin_within_layers[int(argmin_cross_layers)]

            self.importance_dict[cut_layer_name][cut_channel_index] = 100
            if cut_layer_index not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[cut_layer_index] = []
            filters_to_prune_per_layer[cut_layer_index].append(cut_channel_index)
            i += 1

            if pruneTable[cut_layer_index]:
                for sub in pruneTable[cut_layer_index]:
                    if sub not in filters_to_prune_per_layer:
                        filters_to_prune_per_layer[sub] = []
                    filters_to_prune_per_layer[sub].append(cut_channel_index)
                    i += 1

        #print(filters_to_prune_per_layer.keys())
        #for layer,channel in filters_to_prune_per_layer.items():
        #    print("cut_layer_index", layer)
        #    print("cut_channel_index", channel)

        pruned = 0
        cfg = []
        cfg_mask = []
        module_list = list(self.model.modules())
        for index, module in enumerate(self.model.modules()):
            if index==1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                mask = torch.ones(module.weight.shape[0])
                if layer in filters_to_prune_per_layer.keys():
                    for i in filters_to_prune_per_layer[layer]:
                        mask[i] = 0
                mask = mask.cuda()
                ############whole layer is cut
                if int(torch.sum(mask)) == 0:
                    mask[i] = 1.
                #print("mask", mask)

                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(layer, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, nn.MaxPool2d):
                cfg.append('M')

        if 1 not in self.importance_dict.keys():
            cfg[0] = 16
            cfg_mask[0] = cfg_mask[3]
            module = module_list[6]
            module.weight.data.mul_(cfg_mask[0])
            module.bias.data.mul_(cfg_mask[0])
        return pruned, cfg, cfg_mask

num_pruned = int(total * args.percent)
pruner = Pruner(model, num_pruned)
pruned, cfg, cfg_mask = pruner.get_pruning_plan()
print(cfg)
#print(cfg_mask)
#for i in range(len(cfg_mask)):
#    print(len(cfg_mask[i]))
pruned_ratio = pruned/total
print('Pre-processing Successful!')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    print("gate", model.gate)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.item() / len(test_loader.dataset)

def test_time():
        model.eval()
        test_loss = 0
        correct = 0
        duration = 0
        print("gate", model.gate)
        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            start_time = time.time()
            output = model(data)
            duration += time.time() - start_time
            test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), Time: {}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), duration))
        return correct.item()/len(test_loader.dataset)

# new model
def new_cfg():
    mask = [0,0,0]
    index = -1
    count = 0
    n = (config.depth-2) / 9
    for i in range(len(model.gate)):
        if i%n == 0:
            index += 1
        if model.gate[i] == 1:
            count += 1
            mask[index] = (cfg[count*3])
    index = -1
    for i in range(len(model.gate)):
        if i%n == 0:
            index += 1
        if model.gate[i] == 0:
            cfg.insert(i * 3 + 1, mask[index])
            cfg.insert(i * 3 + 1, 0)
            cfg.insert(i * 3 + 1, 0)
    return cfg

#print(cfg)
#print(len(cfg))
#print(model.gate)
channel_cfg  = new_cfg()
#print(channel_cfg)
#print(len(cfg_mask))
#for i in range(len(cfg_mask)):
#    print(len(cfg_mask[i]))
#    print(cfg_mask[i])


def build_new_model():
    new_model = ResNet_v2_channel(depth=config.depth, block_pruning=block_pruning, block_cfg=checkpoint['block_cfg'],
                                  channel_cfg=channel_cfg, num_classes=config.num_classes)
    print(new_model)
    num_parameters = sum([param.nelement() for param in new_model.parameters()])
    savepath = os.path.join(args.save_path, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        #fp.write("Test accuracy: \n"+str(acc))

    downsample_mask = [np.array(0), np.array(0), np.array(0), np.array(0)]
    index = 0
    count = -1
    n = (config.depth - 2) // 9
    for i in range(len(model.gate)):
        if model.gate[i] == 1:
            count += 1
            downsample_mask[index] = cfg_mask[count * 3].cpu().numpy()
        if i % n == 0:
            index += 1
    if not downsample_mask[0].shape:
        downsample_mask[0] = np.ones(16)
    if not downsample_mask[1].shape:
        downsample_mask[1] = np.ones(64)
    if not downsample_mask[2].shape:
        downsample_mask[2] = np.ones(128)
    if not downsample_mask[3].shape:
        downsample_mask[3] = np.ones(256)

    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    down_count = 0
    n = (config.depth-2)//9
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        #print("layer ", layer_id)
        #print("old", m0)
        #print("new", m1)
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
            #print("old", m0.weight.data)
            #print("new", m1.weight.data)
        elif isinstance(m0, nn.Conv2d):
            if isinstance(old_modules[layer_id - 1], nn.BatchNorm2d): # convolutions in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            elif layer_id == 1:
                idx0 = np.squeeze(np.argwhere(start_mask.numpy()))
                idx1 = np.squeeze(np.argwhere(downsample_mask[0]))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            else: # downsampling convolutions.
                idx0 = np.squeeze(np.argwhere(downsample_mask[down_count]))
                idx1 = np.squeeze(np.argwhere(downsample_mask[down_count+1]))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                #print(m1.weight.data.shape)
                down_count += 1
            #print("m0", m0.weight.data)
            #print("m1", m1.weight.data)
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            #print("m0", m0.weight.shape)
            #print("m1", m1.weight.data.shape)
    new_model.gate.data = model.gate.data
    return new_model

def main():
    acc = test()

if __name__ == '__main__':
    main()
    new_model = build_new_model()
    #print(new_model)
    model = new_model.cuda()
    test()
    torch.save({'state_dict': model.state_dict(),
                'block_cfg': model.gate.data,
                'channel_cfg': channel_cfg
                }, os.path.join(args.save_path, 'pruned.pth.tar'))