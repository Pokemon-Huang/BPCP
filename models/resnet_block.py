from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out

    def forward_sss(self, x, gate):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out *= gate
        out += residual

        out = self.relu(out)
        return out

    def forward_fast(self, x, gate):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if gate == 0:
            out = residual
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += residual

        out = self.relu(out)
        return out


class ResNet_block(nn.Module):
    def __init__(self, depth=20, block_pruning=False, block_cfg=None, num_classes=10):
        super(ResNet_block, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
        n = (depth - 2) // 9
        block = Bottleneck

        if block_pruning:
            num_block = n * 3
            print ('block number:', num_block)
            self.alpha = Parameter(torch.zeros(num_block))
            self.gate = Parameter(torch.ones(num_block))  # binary gates
        if block_cfg is None:
            block_cfg = [1] * n * 3

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, n, block_cfg=block_cfg[0: n], stride=1)
        self.layer2 = self._make_layer(block, 32, n, block_cfg=block_cfg[n: n*2], stride=2)
        self.layer3 = self._make_layer(block, 64, n, block_cfg=block_cfg[n*2: n*3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, block_cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        if block_cfg[0]:
            layers.append(block(self.inplanes, planes, stride, downsample))
        else:
            layers.append(downsample)
            layers.append( nn.Sequential( nn.ReLU(inplace=True), ) )
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            if block_cfg[i]:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_prune(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for block in self.layer1:
            if block:
                x = block(x)
        for block in self.layer2:
            if block:
                x = block(x)
        for block in self.layer3:
            if block:
                x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_train(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        num_block = 0
        for block in self.layer1:
            x = block.forward_sss(x, self.gate[num_block])
            num_block += 1
        for block in self.layer2:
            x = block.forward_sss(x, self.gate[num_block])
            num_block += 1
        for block in self.layer3:
            x = block.forward_sss(x, self.gate[num_block])
            num_block += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_test(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        num_block = 0
        for block in self.layer1:
            x = block.forward_fast(x, self.gate[num_block])
            num_block += 1
        for block in self.layer2:
            x = block.forward_fast(x, self.gate[num_block])
            num_block += 1
        for block in self.layer3:
            x = block.forward_fast(x, self.gate[num_block])
            num_block += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_probs(self):
        probs = torch.sigmoid(self.alpha)
        return probs

    def reset_binary_gates(self):
        self.gate.data.zero_()
        probs = self.get_probs()
        sample = torch.bernoulli(probs)
        # set binary gate
        self.gate.data = sample

    def set_test_gates(self):
        self.gate.data.zero_()
        probs = self.get_probs()
        sample = torch.ge(probs, 0.5)
        # set binary gate
        self.gate.data = sample

    def set_alpha_grad(self):
        binary_grads = self.gate.grad.data
        if self.alpha.grad is None:
            self.alpha.grad = torch.zeros_like(self.alpha.data)
        probs = self.get_probs()
        for i in range(self.alpha.data.shape[0]):
            self.alpha.grad.data[i] = binary_grads[i] * probs[i] * (1 - probs[i])

