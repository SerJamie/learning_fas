import sys

sys.path.append("../../")

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from fas import fas_model_fix, fas_model_adapt
import random
import numpy as np
from config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca, configB, configOne
from datetime import datetime
import time
from timeit import default_timer as timer
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import argparse
from PIL import Image
from torchvision import transforms as T
from torch.nn import functional as F
from torch.autograd import Variable
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = "cuda"


def test_one(img_path):
  """
    Eval the final FAS model on the target data.
    python test.py --config [C/M/I/O/cefa/surf/wmca]
    Checkpoints ({casia/msu/replay/oulu/cefa/surf/wmca}_checkpoint.pth.tar) are
    loaded automatically
    """
  # load data
  transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
  img = cv2.imread(img_path)
  img = img.astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  if np.random.randint(2):
    img[..., 1] *= np.random.uniform(0.8, 1.2)
  img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
  img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
  img = transforms(img)

  net = fas_model_adapt(0.3, 0.5).to(device)
  net_ = torch.load("celeb3899_checkpoint.pth.tar")
  net.load_state_dict(net_["state_dict"])
  img = img.unsqueeze(0)
# print(img.shape)
  input = Variable(img).cuda()
  cls_out, feature, total_loss = net(input, True)
  prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
  return prob# #### #### # ####



if __name__ == "__main__":
  img_root = 'test'
  img_paths = os.listdir(img_root)
  img_pr_root = 'proce'
  save_root = 'test_result'

  for i in img_paths:
    img_path = os.path.join(img_root, i)
    img = cv2.imread(img_path)
    results = detector.detect_face(img)
    if results is not None:
       total_boxes = results[0]
       points = results[1]
    # extract aligned face chips
       chips = detector.extract_image_chips(img, points, 256, 0.37)
       for j, chip in enumerate(chips):
          path = os.path.join(img_pr_root, i.split('.')[0] + '_chip_'+str(j)+'.png')
          cv2.imwrite(path, chip)
          prob = test_one(path)

       draw = img.copy()
       for b in total_boxes:
          if(prob >= 0.5):
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
          else:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255))
    save_path = os.path.join(save_root,i.split('.')[0] + 'detection result.png')
    cv2.imwrite(save_path, draw)
    cv2.waitKey(0)

