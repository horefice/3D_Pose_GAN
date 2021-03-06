import numpy as np
import cv2 as cv
import datetime
import os

def color_jet(x):
  if x < 0.25:
    b = 255
    g = x / 0.25 * 255
    r = 0
  elif x >= 0.25 and x < 0.5:
    b = 255 - (x - 0.25) / 0.25 * 255
    g = 255
    r = 0
  elif x >= 0.5 and x < 0.75:
    b = 0
    g = 255
    r = (x - 0.5) / 0.25 * 255
  else:
    b = 0
    g = 255 - (x - 0.75) / 0.25 * 255
    r = 255
  return int(b), int(g), int(r)

def noisy_label(label, factor=0.1):
  noise = factor*torch.rand_like(label)
  noisy_label = (label - noise).abs()

  return noisy_label

def flip_label(label, prob=0.05):
  prob_tensor = torch.zeros_like(label).fill_(prob)
  flip_label = (prob_tensor - torch.bernoulli(prob_tensor)).abs()

  return flip_label

def rotate_and_project(array, z, theta=0):
  x = array[:, 0::2]
  y = array[:, 1::2]
  sin = theta.sin()
  cos = theta.cos()
  
  xx = x*sin + z*sin
  stack = torch.cat((xx,y), dim=-1)

  return stack

def create_projection_img(array, theta=0):
  x = array[:, 0::3]
  y = array[:, 1::3]
  z = array[:, 2::3]

  xx = x * np.cos(theta) + z * np.sin(theta)
  projection = np.stack((xx, y), axis=-1).flatten()

  return create_img(projection)

def create_img(arr, img=None, conf=np.ones(17), thr=0):
  ps = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
  qs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  xs = arr[0::2].copy()
  ys = arr[1::2].copy()
  if img is None:
    xs *= 80
    xs += 100
    ys *= 80
    ys += 150
    xs = xs.astype('i')
    ys = ys.astype('i')
    img = np.zeros((350, 200, 3), dtype=np.uint8) + 160
    img = cv.line(img, (100, 0), (100, 350), (255, 255, 255), 1)
    img = cv.line(img, (0, 150), (200, 150), (255, 255, 255), 1)
    img = cv.rectangle(img, (0, 0), (200, 350), (255, 255, 255), 3)
  for i, (p, q) in enumerate(zip(ps, qs)):
    if (conf[p] < thr) | (conf[q] < thr):
      continue
    c = 1 / (len(ps) - 1) * i
    b, g, r = color_jet(c)
    img = cv.line(img, (xs[p], ys[p]), (xs[q], ys[q]), (b, g, r), 2)
  for i in range(17):
    if conf[i] < thr:
      continue
    c = 1 / 16.0 * i
    b, g, r = color_jet(c)
    img = cv.circle(img, (xs[i], ys[i]), 3, (b, g, r), 3)
  return img

def to36M(bones, body_parts):
  H36M_JOINTS_17 = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
                    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder',
                    'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

  adjusted_bones = []
  for name in H36M_JOINTS_17:
    if not name in body_parts:
      if name == 'Hip':
        bone = (bones[body_parts['RHip']] + 
                               bones[body_parts['LHip']]) / 2
        bone[2] = np.minimum(bones[body_parts['RHip']][2],bones[body_parts['LHip']][2])
        adjusted_bones.append(bone)
      elif name == 'RFoot':
        adjusted_bones.append(bones[body_parts['RAnkle']])
      elif name == 'LFoot':
        adjusted_bones.append(bones[body_parts['LAnkle']])
      elif name == 'Spine':
        bone = (bones[body_parts['RHip']] + bones[body_parts['LHip']] +
                bones[body_parts['RShoulder']] + 
                bones[body_parts['LShoulder']]) / 4
        tmp1 = np.minimum(bones[body_parts['RHip']][2],bones[body_parts['LHip']][2])
        tmp2 = np.minimum(bones[body_parts['RShoulder']][2],bones[body_parts['LShoulder']][2])
        bone[2] = np.minimum(tmp1,tmp2)
        adjusted_bones.append(bone)
      elif name == 'Thorax':
        bone = (bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]) / 2
        bone[2] = np.minimum(bones[body_parts['RShoulder']][2],bones[body_parts['LShoulder']][2])
        adjusted_bones.append(bone)
      elif name == 'Head':
        thorax = (bones[body_parts['RShoulder']] + 
                  bones[body_parts['LShoulder']]) / 2
        thorax[2] = np.minimum(bones[body_parts['RShoulder']][2],bones[body_parts['LShoulder']][2])
        bone = thorax + (bones[body_parts['Nose']] - thorax) * 2
        bone[2] = np.minimum(thorax[2],bones[body_parts['Nose']][2])
        adjusted_bones.append(bone)
      elif name == 'Neck/Nose':
        adjusted_bones.append(bones[body_parts['Nose']])
      else:
        raise Exception(name)
    else:
      adjusted_bones.append(bones[body_parts[name]])

  return adjusted_bones

def normalize_2d(pose):
  # Hip as origin and normalization
  xs = pose.T[0::2] - pose.T[0]
  ys = pose.T[1::2] - pose.T[1]
  pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)

  mu_x = pose[0].copy()
  mu_y = pose[1].copy()
  pose[0::2] -= mu_x
  pose[1::2] -= mu_y
  return pose.T

def flatten_weights(net):
  weights = []
  for layer in net.children():
    for weight in layer.weight.data.cpu().numpy().flatten():
      weights.append(weight)
  return weights

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=10, device='cpu'):
  alpha = torch.rand(real_data.size(0), 1, device=device)
  alpha = alpha.expand(real_data.size())

  interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(device)
  interpolates.requires_grad_()

  output_d = netD(interpolates)

  grads = torch.autograd.grad(outputs=output_d, inputs=interpolates,
                              grad_outputs=torch.ones_like(output_d,
                              device=device), create_graph=True)[0]

  gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
  return gradient_penalty

def writeArgsFile(args,saveDir):
  os.makedirs(saveDir, exist_ok=True)
  args_list = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
  file_name = os.path.join(saveDir, 'args.txt')
  with open(file_name, 'a') as opt_file:
    opt_file.write('\n==> Args ('+datetime.datetime.now().isoformat()+'):\n')
    for k, v in sorted(args_list.items()):
       opt_file.write('  {}: {}\n'.format(str(k), str(v)))