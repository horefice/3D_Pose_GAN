import sys
import os
import numpy as np
import torch
import cv2 as cv
import argparse
import time
import imageio

from posenet import PoseNet
import utils

# PARSER
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--input', type=str, help='Path to image or video. Skip for camera')
parser.add_argument('--proto2d', type=str, help='Path to .prototxt', required=True)
parser.add_argument('--model2d', type=str, help='Path to .caffemodel', required=True)
parser.add_argument('--lift_model', type=str, help='Path to trained 3D model', required=True)
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height')
parser.add_argument('--no_cuda', action='store_true', help='disables CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# GRAPHS (COCO)
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

class OpenPose(object):
    """
    This implementation is based on https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self, proto, model):
        self.net = cv.dnn.readNetFromCaffe(proto, model)

    def predict(self, frame, thr=0.1, width=368, height=368):

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
                                   (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((x, y) if conf > thr else None)
        return points

def create_pose(model, points):
    model.eval()

    x = points[:, 0::2]
    y = points[:, 1::2]

    points = torch.from_numpy(np.array(points))
    if model.is_cuda:
        points = points.cuda()
    z_pred = model.forward(points).data.cpu().numpy()

    pose = np.stack((x, y, z_pred), axis=-1)
    pose = np.reshape(pose, (len(points), -1))

    return pose

def capture_pose(openpose, frame, args):
    points = openpose.predict(frame, args.thr, args.width, args.height)
    points = [np.array(vec) for vec in points]
    points = utils.to36M(points, BODY_PARTS)
    points = np.reshape(points, [1, -1]).astype('f')
    out_img = utils.create_img(points[0], frame)
    
    points_norm = utils.normalize_2d(points)
    pose = create_pose(model, points_norm)

    return pose, out_img

def save_pose(pose, frame, out_directory="output", deg=15):
    os.makedirs(out_directory, exist_ok=True)
    cv.imwrite(os.path.join(out_directory, 'openpose_detect.jpg'), frame)


    images = []
    
    for d in range(0, 360 + deg, deg):
        img = utils.create_projection_img(pose, np.pi * d / 180.)
        images.append(img)
        # cv.imwrite(os.path.join(out_directory, "rot_{:03d}_degree.png".format(d)), img)
    imageio.mimsave(os.path.join(out_directory, "output.gif"),images)
    print("=> Pose saved!")

if __name__ == '__main__':
    openpose = OpenPose(args.proto2d, args.model2d)

    model = PoseNet(mode='generator').to(device)
    if args.lift_model[-4:] == '.npz':
        model.load_npz(args.lift_model)
    else:
        model.load_state_dict(torch.load(args.lift_model)['netG'])

    cap = cv.VideoCapture(args.input if args.input else 0)
    num_frames = 0
    start = time.time()

    while cap.isOpened():
        isFrame, frame = cap.read()
        if not isFrame:
            break
        key = cv.waitKey(1) & 0xFF
        frame = cv.resize(frame, (368, 368))

        if args.input or key == ord('p'):
            pose, frame = capture_pose(openpose, frame, args)
            save_pose(pose, frame)
        else:
            if args.cuda:
                _, frame = capture_pose(openpose, frame, args)
            cv.imshow('Video Demo', frame)

        num_frames += 1
        if key == 27:  # exit
            break

    elasped = time.time() - start
    print("[INFO] elasped time: {:.2f}s".format(elasped))
    print("[INFO] approx. FPS: {:.2f}".format(num_frames / (elasped)))
  
    cap.release()
    cv.destroyAllWindows()
