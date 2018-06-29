import numpy as np
import cv2 as cv
import argparse
import torch
import os

from posenet import PoseNet
from dataHandler import MPII
from viz import Viz
import utils

## SETTINGS
parser = argparse.ArgumentParser()
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
parser.add_argument('--visdom', action='store_true', default=False,
                    help='enables VISDOM')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before saving (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging (default: 10)')

## SETUP
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.saveDir = os.path.join('../models/', args.expID)
utils.writeArgsFile(args,args.saveDir)

device = torch.device('cpu')
torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
  device = torch.device('cuda')
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 1, 'pin_memory': True}

## LOAD DATASETS
print('\nDATASET INFO.')
train_data = MPII('../data/mpii_poses.npy')
print('Train size: {} x {}'.format(len(train_data), train_data[0].size()))

## LOAD MODEL
print('\nLOADING GAN.')

def weights_init(m):
    if type(m) == torch.nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      torch.nn.init.constant_(m.bias, 0.0)

netG = PoseNet(mode='generator').to(device)
netD = PoseNet(mode='discriminator').to(device)
if args.model:
  netG.load_state_dict(torch.load(args.model)['netG'])
  netD.load_state_dict(torch.load(args.model)['netD'])
  print('=> Loaded models from {:s}'.format(args.model))
else:
  netG.apply(weights_init)
  netD.apply(weights_init)
print('Model params: {:.2f}M'.format(sum(p.numel() for p in netG.parameters()) / 1e6))

## TRAINING
print('\nTRAINING.')
data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)

one = torch.FloatTensor([1]).to(device)
mone = (one * -1).to(device)
fixed_data = np.array([ 0.        ,  0.        ,  -0.40023696,  0.04447079,
                       -0.66706157,  0.75600314,  -0.93388623,  0.75600314,
                        0.40023685, -0.04447079,   0.57812   ,  0.57812   ,
                        0.40023685,  1.3785939 ,   0.13341236, -0.5114138 ,
                        0.26682448, -1.0228276 ,   0.31129527, -1.1117692 ,
                        0.35576606, -1.2007108 ,   0.93388605, -1.0228276 ,
                        1.1117692 , -0.31129527,   1.0228276 ,  0.22235394,
                       -0.40023696, -1.0228276 ,  -0.8449447 , -0.40023685,
                       -1.1117692 ,  0.13341236  ])
fixed_t = torch.from_numpy(fixed_data).view(1,-1).float().to(device)

optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr)

iter_per_epoch = len(data_loader)
start_epoch = torch.load(args.model)['epoch'] if args.model else 0

if args.visdom:
  visdom = Viz()
  viz_D = visdom.create_plot('Epoch', 'Loss', 'Loss Discriminator')
  viz_G = visdom.create_plot('Epoch', 'Loss', 'Loss Generator')
  viz_WD = visdom.create_plot('Epoch', 'WD', 'Wasserstein Distance')
  viz_GP = visdom.create_plot('Epoch', 'GP', 'Gradient Penalty')
  viz_img = np.transpose(utils.create_img(fixed_data),(2,0,1))
  viz_img = visdom.create_img(viz_img, title='2D Sample')

print('Start')
for epoch in range(start_epoch, args.epochs):
  for i, data in enumerate(data_loader, 1):
    batch_size = data.size(0)
    real_data = data.to(device)
    angle = 10/180*np.pi # min and (2pi-max) angle for rotation
    
    ############################
    # (1) Update G network: maximize E[D(G(x))]
    ###########################
    netG.zero_grad()
    for p in netD.parameters():
      p.requires_grad = False  # disable to avoid computation
    theta = angle + 2*(np.pi-angle)*torch.rand(batch_size, 17, device=device)
    
    z_pred = netG(real_data)
    fake_data = utils.rotate_and_project(real_data, z_pred, theta)

    G = netD(fake_data)
    G = G.mean()
    G.backward(mone)

    loss_G = -G.item()
    optimizerG.step()

    ############################
    # (2) Update D network: maximize E[D(x)] - E[D(G(x))] + GP
    ###########################
    # train with real
    netD.zero_grad()
    for p in netD.parameters():
      p.requires_grad = True # enable back grad (see below)

    D_real = netD(real_data)
    D_real = D_real.mean()
    D_real.backward(mone)

    # train with fake
    theta = angle + 2*(np.pi-angle)*torch.rand(batch_size, 17, device=device)

    with torch.no_grad():
      z_pred = netG(real_data)
    fake_data = utils.rotate_and_project(real_data, z_pred, theta)

    D_fake = netD(fake_data)
    D_fake = D_fake.mean()
    D_fake.backward(one)

    # gradient penalty
    lambda_gp = 10
    GP = utils.calc_gradient_penalty(netD, real_data, fake_data,
                                     LAMBDA=lambda_gp, device=device)
    GP.backward()
    GP = GP.item()

    loss_D = (D_fake - D_real + GP).item()
    WD = (D_real - D_fake).item()
    optimizerD.step()

    # Log
    if args.log_interval and i % args.log_interval == 0:
      print(('[{:3d}/{:3d}][{:4d}/{:4d}] Loss_D: {:+.4f} Loss_G: {:+.4f} ' +
            'WD: {:+.4f} GP: {:+.4f}')
            .format(epoch+1, args.epochs, i, iter_per_epoch,
                    loss_D, loss_G, WD, GP))

      if args.visdom:
        x = epoch + i / iter_per_epoch
        visdom.update_plot(x=x, y=loss_D, window=viz_D, type_upd='append')
        visdom.update_plot(x=x, y=loss_G, window=viz_G, type_upd='append')
        visdom.update_plot(x=x, y=WD, window=viz_WD, type_upd='append')
        visdom.update_plot(x=x, y=GP, window=viz_GP, type_upd='append')

  # do checkpointing
  if args.save_interval and (epoch+1) % args.save_interval == 0:
    torch.save({'epoch': epoch+1,
               'netG': netG.state_dict(),
               'netD': netD.state_dict()
               }, '{:s}/checkpoint.pth'.format(args.saveDir))
    print('Saved at checkpoint!')

    if args.visdom:
      with torch.no_grad():
        z_pred = netG(fixed_t.unsqueeze(0)).squeeze()
      viz_3D = torch.stack((fixed_t[0::2], fixed_t[1::2], z_pred), dim=1)
      visdom.create_scatter(viz_3D,title='Sample Prediction (epoch {:d})'
                                         .format(epoch+1))

# export model
torch.onnx.export(netG, fixed_t, '{:s}/posenet.proto'.format(args.saveDir))

if args.visdom:
  opts_D = dict(xlabel='Weight', ylabel='Freq', title='Weight Histogram (D)')
  opts_G = dict(xlabel='Weight', ylabel='Freq', title='Weight Histogram (G)')
  visdom.create_hist(utils.flatten_weights(netD), opts=opts_D)
  visdom.create_hist(utils.flatten_weights(netG), opts=opts_G)

print('Finish')
