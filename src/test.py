import sys
sys.path.append('.')

import argparse
import torch

import MinkowskiEngine as ME

from aligner.res16unet import Res16UNet34C

def str2list(l):
  return [int(i) for i in l.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument('--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')
parser.add_argument('--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
parser.add_argument('--bn_momentum', type=float, default=0.02)

config = parser.parse_args()
model = Res16UNet34C(3, 2, config)
# print(model)
x = torch.randn(2, 512, 3)
features = torch.ones(2, 512, 3)
# data = ME.SparseTensor(
#             coordinates=x,
#             features=features,
#             device=torch.device('cuda'))

input_dict = {"coords" : x, "feats" : features}
ME.utils.sparse_collate(**input_dict)

print(data)

# print(model(x).shape)