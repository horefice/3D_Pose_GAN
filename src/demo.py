import numpy as np
import cv2 as cv
import argparse
import time
import sys
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/home/narvis/Lib/openpose/build/python/openpose')
from openpose import *

import onnx, onnx.utils

# PARSER
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--input', type=str, help='Path to image or video. Skip for camera')
parser.add_argument('--model', type=str, help='Path to ONNX model',
                    default='../models/posenet.proto')
parser.add_argument('--backend', type=str, help='ONNX Backend (caffe or tf)',
                    default='tf')
parser.add_argument('--thr', default=0.05, type=float, help='Threshold value for heatmap')
args = parser.parse_args()
if args.backend == 'tf':
  import onnx_tf.backend as backend
elif args.backend == 'caffe':
  import caffe2.python.onnx.backend as backend
else:
  print('[ERROR] A valid backend must be selected!')
  quit()

# GRAPHS (COCO)
BODY_PARTS = {'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4,
              'LShoulder': 5, 'LElbow': 6, 'LWrist': 7, 'RHip': 8, 'RKnee': 9,
              'RAnkle': 10, 'LHip': 11, 'LKnee': 12, 'LAnkle': 13, 'REye': 14,
              'LEye': 15, 'REar': 16, 'LEar': 17, 'Background': 18}

POSE_PAIRS = [['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'],
              ['RElbow', 'RWrist'], ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'],
              ['Neck', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RAnkle'], ['Neck', 'LHip'],
              ['LHip', 'LKnee'], ['LKnee', 'LAnkle'], ['Neck', 'Nose'], ['Nose', 'REye'],
              ['REye', 'REar'], ['Nose', 'LEye'], ['LEye', 'LEar']]

# GRAPHS (H36M)
H36M_JOINTS_17 = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
                  'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder',
                  'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
H36M_POSE_PAIRS = [['Neck/Nose', 'RShoulder'], ['Neck/Nose', 'LShoulder'],
                   ['RShoulder', 'RElbow'], ['RElbow', 'RWrist'],
                   ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'],
                   ['Head','Neck/Nose'], ['Neck/Nose', 'Thorax'],
                   ['Thorax', 'Spine'], ['Spine', 'Hip'],
                   ['Hip', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RFoot'],
                   ['Hip', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LFoot']]

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

def create_projection_img(array, r1=0, r2=0):
  x = array[:, 0::3]
  y = array[:, 1::3]
  z = array[:, 2::3]

  rotMat, _ = cv.Rodrigues(np.array([r2,r1,0.]))
  projection = np.dot(rotMat, np.concatenate((x,y,z),0))
  projection = np.stack((projection[0]/projection[2], projection[1]/projection[2]), axis=-1).flatten()
  #print(projection)

  return create_img(projection)

def create_img(arr, img=None):
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
    c = 1 / (len(ps) - 1) * i
    b, g, r = color_jet(c)
    img = cv.line(img, (xs[p], ys[p]), (xs[q], ys[q]), (b, g, r), 2)
  for i in range(17):
    c = 1 / 16 * i
    b, g, r = color_jet(c)
    img = cv.circle(img, (xs[i], ys[i]), 3, (b, g, r), 3)
  return img

def to36M(bones):
  adjusted_bones = []
  for name in H36M_JOINTS_17:
    if not name in BODY_PARTS:
      if name == 'Hip':
        adjusted_bones.append((bones[BODY_PARTS['RHip']] + 
                               bones[BODY_PARTS['LHip']]) / 2)
      elif name == 'RFoot':
        adjusted_bones.append(bones[BODY_PARTS['RAnkle']])
      elif name == 'LFoot':
        adjusted_bones.append(bones[BODY_PARTS['LAnkle']])
      elif name == 'Spine':
        adjusted_bones.append(
          (bones[BODY_PARTS['RHip']] + bones[BODY_PARTS['LHip']] +
          bones[BODY_PARTS['RShoulder']] + 
          bones[BODY_PARTS['LShoulder']]) / 4)
      elif name == 'Thorax':
        adjusted_bones.append(
          (bones[BODY_PARTS['RShoulder']] + 
           bones[BODY_PARTS['LShoulder']]) / 2)
      elif name == 'Head':
        thorax = (bones[BODY_PARTS['RShoulder']] + 
                  bones[BODY_PARTS['LShoulder']]) / 2
        adjusted_bones.append(thorax + 
                              (bones[BODY_PARTS['Nose']] - thorax) * 2)
      elif name == 'Neck/Nose':
        adjusted_bones.append(bones[BODY_PARTS['Nose']])
      else:
        raise Exception(name)
    else:
      adjusted_bones.append(bones[BODY_PARTS[name]])

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

def capture_pose(frame):
  points, frame_op = openpose.forward(frame, display=True)

  if len(points) == 0:
    return [], [], frame_op

  points = [np.array(vec) for vec in points[0]]
  points = to36M(points)
  points = np.reshape(points, -1)

  conf = points[2::3]
  points = np.delete(points, np.arange(2, points.shape[0], 3))
  points = np.reshape(points, (1, -1))
  
  points = normalize_2d(points)

  z_pred = np.zeros([1,17])
  #posenet.run(points)

  pose = np.stack((points[:, 0::2], points[:, 1::2], z_pred), axis=-1)
  pose = np.reshape(pose, (pose.shape[0], -1))

  return pose, conf, frame_op

def frame_to_3D(frame, pose, conf):
  width = 184
  height = 184

  a = create_projection_img(pose, 0, 0)
  a = cv.resize(a, (height, width))
  b = create_projection_img(pose, 90, 0)
  b = cv.resize(b, (height, width))
  vstack1 = np.vstack((a,b))

  c = create_projection_img(pose, 45, 45)
  c = cv.resize(c, (height, width))
  d = create_projection_img(pose, 0,90)
  d = cv.resize(d, (height, width))
  vstack2 = np.vstack((c,d))

  list_conf = np.zeros((height*2,width,3), np.float)
  thr = 0.1
  font = cv.FONT_HERSHEY_SIMPLEX
  line = cv.LINE_AA

  for i,joint in enumerate(H36M_JOINTS_17,0):
    spacing = (5,30+i*20)

    color = (255,255,255)
    if conf[0,i] < thr:
      color = (0,0,255)

    cv.putText(img_conf, joint + ' {:.3f}'.format(conf[0,i]), spacing,
               font, .6, color, 1, line)

  hstack = np.hstack((vstack1, frame, vstack2, list_conf))
  return hstack

def display_pose(pose):
  f = plt.figure()
  f.suptitle('Demo')

  ax = f.add_subplot(111, projection='3d')
  ax.set_title('3D Pose Estimation')

  for part in H36M_JOINTS_17:
    i = H36M_JOINTS_17.index(part)*3
    ax.plot([pose[i]],[pose[i+1]],[pose[i+2]], 'o')
  for pair in H36M_POSE_PAIRS:
    i1 = H36M_JOINTS_17.index(pair[0])*3
    i2 = H36M_JOINTS_17.index(pair[1])*3
    ax.plot([pose[i1],pose[i2]],
          [pose[i1+1],pose[i2+1]],
          [pose[i1+2],pose[i2+2]], 'g')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.view_init(-90,-90)

  plt.show()

if __name__ == '__main__':
  # OpenPose
  params = dict()
  params["logging_level"] = 3
  params["output_resolution"] = "-1x-1"
  params["net_resolution"] = "-1x368"
  params["model_pose"] = "COCO"
  params["alpha_pose"] = 0.6
  params["scale_gap"] = 0.3
  params["scale_number"] = 1
  params["render_threshold"] = args.thr
  params["num_gpu_start"] = 0
  params["disable_blending"] = False
  params["default_model_folder"] = "/home/narvis/Lib/openpose/models/"
  openpose = OpenPose(params)

  model = onnx.load(args.model)
  model = onnx.utils.polish_model(model)
  posenet = backend.prepare(model)
  print('=> Models loaded!')

  cap = cv.VideoCapture(args.input if args.input else 0)
  pose = []
  conf = []
  num_frames = 0
  time_paused = 0
  start = time.time()

  while cap.isOpened():
    isFrame, frame = cap.read()
    if not isFrame:
      break
    key = cv.waitKey(1) & 0xFF
    frame = cv.resize(frame, (368, 368))

    pose_tmp, conf_tmp, frame = capture_pose(frame)
    if len(pose_tmp) > 0:
      pose, conf = pose_tmp, conf_tmp

    if key == ord('p'):  # pause
      start_pause = time.time()

      while True:
        key2 = cv.waitKey(1) or 0xff
        cv.imshow('Video Demo', frame)
        if key2 == ord('p'):  # resume
          time_paused += time.time() - start_pause
          break

    cv.imshow('Video Demo', frame)

    num_frames += 1
    if key == 27:  # exit
      break

  elasped = time.time() - start
  print('[INFO] elasped time: {:.2f}s'.format(elasped))
  print('[INFO] approx. FPS: {:.2f}'.format(num_frames / (elasped-time_paused)))
  
  cap.release()
  cv.destroyAllWindows()

  if len(pose) > 0:
    display_pose(pose[0])
  else:
    print('[ERROR] No pose detected!')
