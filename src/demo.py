import numpy as np
import cv2 as cv
import argparse
import time
import sys
import os

import webCamUtils
from utils import color_jet, create_img, to36M, normalize_2d

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
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for rendering')
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

# GRAPHS (H36M)
H36M_JOINTS_17 = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
                  'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder',
                  'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
H36M_POSE_PAIRS = [['Hip', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RFoot'],
                   ['Hip', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LFoot'],
                   ['Hip', 'Spine'],['Spine', 'Thorax'],
                   ['Thorax', 'Neck/Nose'], ['Neck/Nose','Head'],
                   ['Thorax', 'LShoulder'], ['LShoulder', 'LElbow'],
                   ['LElbow', 'LWrist'], ['Thorax', 'RShoulder'],
                   ['RShoulder', 'RElbow'], ['RElbow', 'RWrist']]

def create_projection_img(array, r1=0, r2=0, conf=[], thr=0):
  x = array[:, 0::3]
  y = array[:, 1::3]
  z = array[:, 2::3]

  camera = np.array([[1,0,0],[0,1,0],[0,0,1]])
  tMat = np.zeros((3,1))
  rMat, _ = cv.Rodrigues((r2*np.pi/180,r1*np.pi/180,0.))
  pMat = camera.dot(np.hstack((rMat,tMat)))

  projection = np.dot(pMat, np.concatenate((x,y,z,np.ones(x.shape)),0))
  projection = np.stack((projection[0], projection[1]), axis=-1).flatten()

  return create_img(projection, conf=conf, thr=thr)

def capture_pose(frame):
  points, frame_op = openpose.forward(frame, display=True)

  if len(points) == 0:
    return [], [], frame_op

  points = [np.array(vec) for vec in points[0]]
  points = to36M(points, BODY_PARTS)
  points = np.reshape(points, -1)

  conf = points[2::3]
  points = np.delete(points, np.arange(2, points.shape[0], 3))

  frame = create_img(points, img=frame, conf=conf, thr=args.thr)

  points = np.reshape(points, (1, -1))
  points = normalize_2d(points)

  #z_pred = np.tanh(np.random.randn(1,17))
  z_pred = posenet.run(np.concatenate((points,points), axis=0))._0[0,:].reshape(1,-1)

  pose = np.stack((points[:, 0::2], points[:, 1::2], z_pred), axis=-1)
  pose = np.reshape(pose, (pose.shape[0], -1))

  return pose, conf, frame

def frame_to_GUI(frame, pose, conf):
  width = 184
  height = 184
  sideAngle = 45
  elevation = 45
  views = [(sideAngle, elevation),(sideAngle, 0),
           (-sideAngle, elevation),(-sideAngle, 0)]

  font = cv.FONT_HERSHEY_SIMPLEX
  line = cv.LINE_AA
  spacing = (5, height - 5)
  fontscale = .5
  color = (255, 255, 255)
  thickness = 1

  a = create_projection_img(pose, views[0][0], views[0][1], conf, args.thr)
  a = cv.resize(a, (height, width))
  cv.putText(a, "SideRot:{:d} Elev:{:d}".format(views[0][0], views[0][1]), spacing, font, fontscale, color, thickness, line)
  b = create_projection_img(pose, views[1][0], views[1][1], conf, args.thr)
  b = cv.resize(b, (height, width))
  cv.putText(b, "SideRot:{:d} Elev:{:d}".format(views[1][0], views[1][1]), spacing, font, fontscale, color, thickness, line)
  vstack1 = np.vstack((a,b))

  c = create_projection_img(pose, views[2][0], views[2][1], conf, args.thr)
  c = cv.resize(c, (height, width))
  cv.putText(c, "SideRot:{:d} Elev:{:d}".format(views[2][0], views[2][1]), spacing, font, fontscale, color, thickness, line)
  d = create_projection_img(pose, views[3][0], views[3][1], conf, args.thr)
  d = cv.resize(d, (height, width))
  cv.putText(d, "SideRot:{:d} Elev:{:d}".format(views[3][0], views[3][1]), spacing, font, fontscale, color, thickness, line)
  vstack2 = np.vstack((c,d))

  img_conf = np.zeros((height*2,width,3), np.uint8)

  for i,joint in enumerate(H36M_JOINTS_17):
    spacing = (5,30+i*20)
    conf[i] = min(max(0,conf[i]),1)

    if conf[i] < args.thr:
      color_conf = (0,0,255)
    else:
      color_conf = color

    cv.putText(img_conf, joint + ' {:.3f}'.format(conf[i]), spacing,
               font, fontscale, color_conf, thickness, line)

  hstack = np.hstack((vstack1, frame, vstack2, img_conf))
  return hstack

def display_pose(pose, conf=np.ones(17), thr=0):
  f = plt.figure()
  f.suptitle('Demo')

  ax = f.add_subplot(111, projection='3d')
  ax.set_title('3D Pose Estimation')

  for j, part in enumerate(H36M_JOINTS_17):
    if conf[j] < thr:
      continue
    i = H36M_JOINTS_17.index(part)*3
    ax.plot([pose[i]],[pose[i+1]],[pose[i+2]], 'o')
  for j, pair in enumerate(H36M_POSE_PAIRS):
    i1 = H36M_JOINTS_17.index(pair[0])*3
    i2 = H36M_JOINTS_17.index(pair[1])*3
    if (conf[int(i1/3)] < thr) | (conf[int(i2/3)] < thr):
      continue
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

  cap = webCamUtils.WebcamVideoStream(args.input if args.input else 0)
  #cap = cv.VideoCapture(args.input if args.input else 0)

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

    frame = frame_to_GUI(frame, pose, conf)

    if key == ord('p'): # pause
      start_pause = time.time()

      while True:
        key2 = cv.waitKey(1) or 0xff
        cv.imshow('Demo', frame)
        if key2 == ord('p'): # resume
          time_paused += time.time() - start_pause
          break
        elif key2 == 27: # exit
          time_paused += time.time() - start_pause
          key = 27
          break

    cv.imshow('Demo', frame)

    num_frames += 1
    if key == 27: # exit
      break
  else:
    start_pause = time.time()
    cv.waitKey(0)
    time_paused += time.time() - start_pause

  elasped = time.time() - start
  print('[INFO] elasped time: {:.2f}s'.format(elasped))
  print('[INFO] approx. FPS: {:.2f}'.format(num_frames / (elasped-time_paused)))
  
  cap.release()
  cv.destroyAllWindows()

  if len(pose) > 0:
    display_pose(pose[0], conf=conf, thr=args.thr)
  else:
    print('=> No pose detected!')
