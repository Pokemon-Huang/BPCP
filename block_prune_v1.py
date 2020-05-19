from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time
import shutil
import argparse
from models import *
import config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_path', default='./exp/block_prune', type=str, metavar='PATH',
                        help='path to save dir (default: none)')
    parser.add_argument('--prune', default='./exp/pretrained/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to pretrained model (default: none)')
    return parser.parse_args()

# initial
block_pruning = True
channel_pruning = False
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
model = ResNet_block(depth=config.depth, block_pruning=block_pruning,
                                        num_classes=config.num_classes)
checkpoint = torch.load(args.prune)
model.load_state_dict(checkpoint['state_dict'], strict=False)
if use_cuda:
    model.cuda()
print(model)

base_param = []
for pname, p in model.named_parameters():
    if pname!='alpha':
        base_param += [p]
base_optimizer = optim.SGD([{'params': base_param, 'weight_decay': config.weight_decay}],
                           lr=config.lr, momentum=config.momentum)
arch_optimizer = optim.SGD([{'params': model.alpha, 'weight_decay': 0}],
                           lr=config.lr, momentum=config.momentum)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        config.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        base_optimizer.load_state_dict(checkpoint['base_optimizer'])
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
#config.start_epoch = 80

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        base_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        model.reset_binary_gates()
        output = model.forward_train(data)
        loss = F.cross_entropy(output, target)
        l1_loss = config.gamma * F.l1_loss(model.gate, torch.zeros(model.gate.size()).cuda(),
                                         reduction='sum')
        loss += l1_loss
        loss.backward()
        model.set_alpha_grad()
        base_optimizer.step()
        arch_optimizer.step()
        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    model.set_test_gates()
    print("alpha", model.alpha)
    print("gate", model.gate)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model.forward_test(data)
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
    model.set_test_gates()
    print("alpha", model.alpha)
    print("gate", model.gate)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        start_time = time.time()
        output = model.forward_test(data)
        duration += time.time() - start_time
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), Time: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), duration))
    return correct.item()/len(test_loader.dataset)

def test_prune():
    model.eval()
    test_loss = 0
    correct = 0
    model.set_test_gates()
    print("alpha", model.alpha)
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

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))

def build_new_model():
    new_model = ResNet_block(depth=config.depth, block_pruning=block_pruning,
                                block_cfg = model.gate, num_classes=config.num_classes)

    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    old_bottlenecks = []
    old_downsamples = []
    index = -1
    for layer in range(len(old_modules)):
        if isinstance(old_modules[layer], Bottleneck_v1):
            index += 1
            if model.gate[index] == 1:
                old_bottlenecks.append(old_modules[layer])
        if layer != 1 and isinstance(old_modules[layer], nn.Conv2d) and not isinstance(old_modules[layer + 1], nn.BatchNorm2d):
            old_downsamples.append(old_modules[layer])
    ib = -1
    id = -1
    for layer, module in enumerate(new_model.modules()):
        if isinstance(module, Bottleneck_v1):
            ib += 1
            sub0_modules = list(old_bottlenecks[ib].modules())
            sub1_modules = list(module.modules())
            for layer_id in range(8):
                m0 = sub0_modules[layer_id]
                m1 = sub1_modules[layer_id]
                if isinstance(m0, nn.Conv2d):
                    m1.weight.data = m0.weight.data.clone()
                elif isinstance(m0, nn.BatchNorm2d):
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                    m1.running_mean = m0.running_mean.clone()
                    m1.running_var = m0.running_var.clone()
        elif isinstance(module, nn.Conv2d):
            if layer == 1: # head conv
                module.weight.data = old_modules[1].weight.data.clone()
            elif not isinstance(new_modules[layer + 1], nn.BatchNorm2d): # down sample
                id += 1
                module.weight.data = old_downsamples[id].weight.data.clone()

    new_model.alpha.data = model.alpha
    new_model.gate.data = model.gate

    modules = list(model.modules())
    new_modules = list(new_model.modules())
    # fc
    m0 = modules[-1]
    m1 = new_modules[-1]
    m1.weight.data = m0.weight.data.clone()
    m1.bias.data = m0.bias.data.clone()
    # bn1
    m0 = modules[2]
    m1 = new_modules[2]
    m1.weight.data = m0.weight.data.clone()
    m1.bias.data = m0.bias.data.clone()
    m1.running_mean = m0.running_mean.clone()
    m1.running_var = m0.running_var.clone()

    return new_model

def main():
    if args.test:
        test_time()
        return 0
    test()
    best_prec1 = 0.
    for epoch in range(config.start_epoch, config.epoch):
        if epoch in [config.epoch*0.5, config.epoch*0.75]:
            for param_group in base_optimizer.param_groups or arch_optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'base_optimizer': base_optimizer.state_dict(),
            'arch_optimizer': arch_optimizer.state_dict(),
        }, is_best, filepath=args.save_path)


if __name__ == '__main__':
    main()
    new_model = build_new_model()
    model = new_model.cuda()
    test_prune()

    torch.save({'state_dict': model.state_dict(),
                'block_cfg': model.gate.data,
                }, os.path.join(args.save_path, 'pruned.pth.tar'))