import argparse
import os
import numpy as np
from utils import processor
from dataprocess import loader
from net import classifier
import torch.nn as nn
import torch
import torchlight

# Loader class parameters
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')
ftype = ''

# Path for saved models
# model_path = os.path.join(base_path, 'model_classifier_stgcn/features2D'+ftype)

#%% - Parsing Arguments
parser = argparse.ArgumentParser(description='Depth prediction')
# Arguments for training the model - Default value

# Training - True
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
# Saving the features - True. Currently disabled
parser.add_argument('--save-features', type=bool, default=True, metavar='SF',
                    help='save penultimate layer features (default: True)')

# Batch size - 8
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')

# Number of Workers - 4
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')

# Starting epoch for training - 0
parser.add_argument('--start_epoch', type=int, default=1, metavar='SE',
                    help='starting epoch of training (default: 0)')

# Number of epochs - 500
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')

# Optimizer - Adam
parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                    help='optimizer (default: SGD)')

# Learning rate - 0.1
parser.add_argument('--base-lr', type=float, default=0.01, metavar='L',
                    help='base learning rate (default: 0.01)')

# Modification of learning rate steps - [0.5, 0.75, 0.875]. Currently not adjusting the learning rate
parser.add_argument('--step', type=int, default=6, metavar='ST',
                    help='iteration step after which the learning rate is reduced by a fraction of 0.5')

# Learning rate threshold
parser.add_argument('--lr_thresh', type=int, default=0.01, metavar='LT',
                    help='learning rate threshold')

# Nesterov - Dunno what is this, branched of the parent code
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')

# Momentum rate - 0.9
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')

# Weight-decay rate - 5*10^-4
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')

# Evaluation interval - 1
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')

# Logging interval - 100
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')

# Topk [1]. Currently disabled
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')

# No-cuda - False
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# Print Log - True
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')

# save Log - True
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')

# Working directory - model_path. Currently disabled
# parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
#                     help='path to save')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

#%% TBD: Load the dataset
# data, labels, data_train, labels_train, data_test, labels_test = \
    # loader.load_data(data_path, ftype, coords, joints, cycles=cycles)
# num_classes = np.unique(labels_train).shape[0]
# dataset_object = loader.TrainTestLoader(type = "train")
# print("Total dataset length: ", dataset_object.__len__())
train_dataset_object = loader.TrainTestLoader(type = "train")
eval_dataset_object = loader.TrainTestLoader(type = "val")
test_dataset_object = loader.TrainTestLoader(type = "test")
print("train length: ", len(train_dataset_object))
print("eval length: ", len(eval_dataset_object))
print("test length: ", len(test_dataset_object))

# print("Train Sample Data shape, train sample label shape: ",dataset_object[2][0].shape, dataset_object[2][1].shape)
data_loader_train_test = list()
data_loader_train_test.append(torch.utils.data.DataLoader(
    dataset=train_dataset_object, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_worker * torchlight.ngpu(device),
    drop_last=False))
data_loader_train_test.append(torch.utils.data.DataLoader(
    dataset=eval_dataset_object, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_worker * torchlight.ngpu(device),
    drop_last=False))

data_loader_train_test.append(torch.utils.data.DataLoader(
    dataset=test_dataset_object, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_worker * torchlight.ngpu(device),
    drop_last=False))

data_loader_train_test = dict(train=data_loader_train_test[0], validation=data_loader_train_test[1],
                              test=data_loader_train_test[2])



pr = processor.Processor(args, data_loader_train_test, device=device)

if args.train:
    pr.train()
else:
    pr.test()
