import numpy as np
import cv2 as cv
import argparse
import torch
import os

from posenet import PoseNet
from dataHandler import MPII
import utils

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet Implementation')
parser.add_argument('-x', '--expID', type=str, default='test', metavar='S',
                    help='Experiment ID')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', '--learning-rate', type=float, default=2e-4, metavar='F',
                    help='learning rate (default: 2e-4)')
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='path to saved model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging (default: 10)')

## SETUP
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.saveDir = os.path.join('../models/', args.expID)

device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 1, 'pin_memory': True}

os.makedirs(args.saveDir, exist_ok=True)
utils.writeArgsFile(args,args.saveDir)

## LOAD DATASETS
print('\nDATASET INFO.')
train_data = MPII('../data/mpii_poses.npy')
print('Train size: {} x {}'.format(len(train_data), train_data[0].size()))

## LOAD MODEL
print('\nLOADING GAN.')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      # m.weight.data.normal_(0.0, 0.02)
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

netG = PoseNet(mode="generator").to(device)
netG.apply(weights_init)
netD = PoseNet(mode="discriminator").to(device)
netD.apply(weights_init)
if args.model:
  netG.load_state_dict(torch.load(args.model)['netG'])
  netD.load_state_dict(torch.load(args.model)['netD'])
  print("=> Loaded models from {}".format(args.model))
print("Model params: {:.2f}M".format(sum(p.numel() for p in netG.parameters()) / 1000000.))

## TRAINING
print('\nTRAINING.')
data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)

fixed_noise = torch.randn(args.batch_size, 34, device=device)
real_label = 1
fake_label = 0

criterion = torch.nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr)

iter_per_epoch = len(data_loader)
start_epoch = torch.load(args.model)['epoch'] if args.model else 0
skip_d_update = False

print('Start')
for epoch in range(start_epoch, args.epochs):
  for i, data in enumerate(data_loader, 0):
    batch_size = data.size(0)
    real_data = data.to(device)
    label = torch.full((batch_size,1), real_label, device=device)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    if not skip_d_update:
      netD.zero_grad()

      output = netD(real_data)

      errD_real = criterion(output, label)
      errD_real.backward()

      D_x = output.mean().item()

      # train with fake_data
      noise = torch.randn(batch_size, 34, device=device)
      deg = np.random.uniform(1,360)

      gen_data = netG(noise).detach()
      fake_data = utils.stack_and_project(noise, gen_data, np.pi * deg / 180.)
      fake_data = torch.from_numpy(fake_data)
      label.fill_(fake_label)
      output = netD(fake_data)

      errD_fake = criterion(output, label)
      errD_fake.backward()

      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(f(p,G(p),theta)))
    ###########################
    netG.zero_grad()
    deg = np.random.uniform(1,360)

    gen_data = netG(real_data).detach()
    fake_data = utils.stack_and_project(real_data, gen_data, np.pi * deg / 180.)
    fake_data = torch.from_numpy(fake_data)
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake_data)

    errG = criterion(output, label)
    errG.backward()

    D_G_z2 = output.mean().item()
    optimizerG.step()

    skip_d_update = False#D_G_z2 > 0.9

    # Log
    if args.log_interval and i % args.log_interval == 0:
      print(('[{:2d}/{:2d}][{:3d}/{:3d}] Loss_D: {:.4f} Loss_G: {:.4f} ' +
            'D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}')
            .format(epoch, args.epochs, i, iter_per_epoch,
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

  # do checkpointing
  print('Saving at checkpoint...')
  torch.save({'epoch': epoch+1,
             'netG': netG.state_dict(),
             'netD': netD.state_dict()
             }, '{:s}/checkpoint.pth'.format(args.saveDir))
print('Finish')
