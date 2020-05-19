from __future__ import absolute_import

from .resnet import ResNet
from .resnet_v2 import ResNet_v2
from .resnet_block import ResNet_block
from .resnet_v2_block import ResNet_v2 as ResNet_v2_block
from .resnet_v2_channel import ResNet_v2 as ResNet_v2_channel
from .resnet_block import Bottleneck as Bottleneck_v1
from .resnet_v2_block import Bottleneck as Bottleneck_block
from .resnet_v2_channel import Bottleneck as Bottleneck_channal

__all__ = ['ResNet', 'ResNet_v2', 'ResNet_block', 'ResNet_v2_block', 'ResNet_v2_channel', 'Bottleneck_v1', 'Bottleneck_block', 'Bottleneck_channal']