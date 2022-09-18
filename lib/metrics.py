from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]
    Returns:
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

# given the predicted R|T, and the ground-truth R|T
# computing the ADD distance for the specified object model
# model_pts: (3, N)
# predict_RT: (4, 4)
# gt_RT: (4, 4)
def compute_add(predict_RT, gt_RT, model_pts):
    # predict pts
    predict_pts = transform_coordinates_3d(model_pts, predict_RT) # (3, N)
    # gt pts
    gt_pts = transform_coordinates_3d(model_pts, gt_RT) # (3, N)

    predict_pts = torch.from_numpy(predict_pts).cuda()
    gt_pts = torch.from_numpy(gt_pts).cuda()

    dis = torch.mean(torch.norm((predict_pts - gt_pts), dim=0))

    return dis.cpu().item()

# given the predicted R|T, and the ground-truth R|T
# computing the ADD-S distance for the specified object model
# model_pts: (3, N)
# predict_RT: (4, 4)
# gt_RT: (4, 4)
def compute_adds(predict_RT, gt_RT, model_pts):
    # predict pts
    predict_pts = transform_coordinates_3d(model_pts, predict_RT) # (3, N)
    # gt pts
    gt_pts = transform_coordinates_3d(model_pts, gt_RT) # (3, N)

    predict_pts = torch.from_numpy(predict_pts).cuda()
    gt_pts = torch.from_numpy(gt_pts).cuda()

    k_nearest = 1

    # _, ind = knn(predict_pts.unsqueeze(0), gt_pts.unsqueeze(0))
    ind = KNearestNeighbor.apply(predict_pts.unsqueeze(0), gt_pts.unsqueeze(0),1)
    reference_pts = torch.index_select(predict_pts, 1, ind.view(-1) - 1)

    dis = torch.mean(torch.norm((reference_pts - gt_pts), dim=0))

    return dis.cpu().item()

# model_pts: (3, N)
# predict_RT: (B, 4, 4)
# gt_RT: (B, 4, 4)
def batch_compute_add(predict_RT, gt_RT, model_pts):
    add_list = []
    number_samples = predict_RT.shape[0]
    for i in range(number_samples):
        dis = compute_add(predict_RT[i], gt_RT[i], model_pts)
        # print(dis)
        add_list.append(dis)
    
    return add_list
    
# model_pts: (3, N)
# predict_RT: (B, 4, 4)
# gt_RT: (B, 4, 4)
def batch_compute_adds(predict_RT, gt_RT, model_pts):
    adds_list = []
    number_samples = predict_RT.shape[0]
    for i in range(number_samples):
        dis = compute_adds(predict_RT[i], gt_RT[i], model_pts)
        adds_list.append(dis)
    
    return adds_list

def compute_accuracy(dis_list, threshold):
    correct_num = np.sum(np.array(dis_list) < threshold)
    return correct_num / len(dis_list)

def compute_auc(dis_list, max_threshold):
    step = max_threshold / 100
    thresholds = np.arange(0, max_threshold+step, step)

    auc = 0.0
    for i in range(len(thresholds)):
        auc += (compute_accuracy(dis_list, thresholds[i]) * step)
    
    return auc * (100.0 / max_threshold)

def test_knn():
    for i in range(1000):
        print(i)
        ref = torch.rand((3, 2000)).cuda()
        query = torch.rand((3, 2000)).cuda()

        k_nearest = 1
        ind = KNearestNeighbor.apply(ref.unsqueeze(0), query.unsqueeze(0),k_nearest)
        # print(ind)

if __name__ == '__main__':
    test_knn()
