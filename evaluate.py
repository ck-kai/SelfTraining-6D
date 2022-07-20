import os
import cv2
import time
import sys
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))

import argparse
from PIL import Image
import numpy as np
import copy
import _pickle as cPickle
import numpy.ma as ma
import scipy.io as scio
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.transformations import quaternion_matrix
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.metrics import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--obj_dataset', type=str, default='robi')
parser.add_argument('--obj_name', type=str, default='zigzag')
parser.add_argument('--testing_mode', type=str, default='virtual', help='specify the model used for testing, should be [virtual, st]')
parser.add_argument('--testing_iter', type=int, default=1, help='number of iterations of the model used for testing')
parser.add_argument('--result_dir', type=str, default='./results')

opt = parser.parse_args()
norm = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                                std=[0.229, 0.229, 0.229])])
intrinsics = np.identity(3, np.float32)

opt.img_height = 1024
opt.img_width = 1280

opt.cam_cx = 379.32687
opt.cam_cy = 509.43720
opt.cam_fx = 1083.09705
opt.cam_fy = 1083.09705

xmap = np.array([[i for i in range(opt.img_width)] for j in range(opt.img_height)])
ymap = np.array([[j for i in range(opt.img_width)] for j in range(opt.img_height)])
intrinsics[0, 0] = opt.cam_fx
intrinsics[1, 1] = opt.cam_fy
intrinsics[0, 2] = opt.cam_cx
intrinsics[1, 2] = opt.cam_cy

img_patch_size = 192
depth_scale = 1000.0
num_obj = 1
num_points = 1000
num_rotations = 60

def get_bbox(bbox, img_height, img_width):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = img_height
    img_length = img_width

    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 640)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def inference(obj_name, testing_mode, testing_iter, result_dir):
    dataset_config_dir = os.path.join('./dataset', obj_name, 'dataset_config')
    data_dir = os.path.join('./data', obj_name, 'testing_data')

    if testing_mode == 'virtual':
        experiment_name = testing_mode
    else:
        experiment_name = testing_mode + '_' + str('%02d' % testing_iter)

    current_result_dir = os.path.join(result_dir, obj_name, experiment_name)
    if not os.path.exists(current_result_dir):
        os.makedirs(current_result_dir)
    
    # generate the test file list, and load the cad model
    filelist = []
    test_list_file = open('{0}/real_test_list.txt'.format(dataset_config_dir))
    while 1:
        test_line = test_list_file.readline()
        if not test_line:
            break
        if test_line[-1:] == '\n':
            test_line = test_line[:-1]
        filelist.append(test_line)
    test_list_file.close()
    print('there are totally {0} scenes to test'.format(len(filelist)))
    cad_model = np.load('{0}/cad.npy'.format(dataset_config_dir))

    # load the trained network
    if testing_mode == 'virtual':
        model_path = os.path.join('./virtual_models', obj_name, 'pose_model.pth')
    else:
        model_dir = os.path.join('./real_models', obj_name)
        for i in range(testing_iter, 0, -1):
            current_model_dir = os.path.join(model_dir, str('%02d' % i))
            model_path = os.path.join(current_model_dir, 'pose_model.pth')
            if os.path.exists(model_path):
                break

    estimator = PoseNet(num_points=num_points, num_obj=num_obj, num_rot=num_rotations)
    estimator.cuda()
    estimator.load_state_dict(torch.load(model_path))
    estimator.eval()

    overall_RT_predict = {}
    overall_RT_predict_icp = {}
    overall_RT_gt = {}

    # traverse all data for testing
    for idx in range(len(filelist)):
        print(filelist[idx])
        # load data corresponding to current scene
        img = cv2.imread('{0}/{1}_color.png'.format(data_dir, filelist[idx]))[:, :, :3]
        img = img[:, :, ::-1]
        depth = np.array(cv2.imread('{0}/{1}_depth.png'.format(data_dir, filelist[idx]), -1))
        mask = np.array(cv2.imread('{0}/{1}_mask.png'.format(data_dir, filelist[idx]), -1))
        with open('{0}/{1}_label.pkl'.format(data_dir, filelist[idx]), 'rb') as f:
            label = cPickle.load(f)
        
        # store results for current frame
        RT_predict = []
        RT_gt = []
        # trverse all instances
        for iid in range(len(label['instance_ids'])):
            # note that should plus one
            inst_id = label['instance_ids'][iid] + 1
            rmin, rmax, cmin, cmax = get_bbox(label['bboxes'][iid], opt.img_height, opt.img_width)

            # sample points
            current_mask = np.equal(mask, inst_id)
            current_mask = np.logical_and(current_mask, depth>0)
            choose = current_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            # gt pose
            target_r = label['rotations'][iid]
            target_t = label['translations'][iid]

            # only test the instance whose number of observed points is larger than 100
            if len(choose) > 100:
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                pt2 = depth_masked / depth_scale
                pt0 = (xmap_masked - opt.cam_cx) * pt2 / opt.cam_fx
                pt1 = (ymap_masked - opt.cam_cy) * pt2 / opt.cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                
                import copy
                points = copy.deepcopy(cloud)

                # resize cropped image to standard size and adjust 'choose' accordingly
                img_masked = copy.deepcopy(img[rmin:rmax, cmin:cmax, :])
                img_masked = cv2.resize(img_masked, (img_patch_size, img_patch_size), interpolation=cv2.INTER_LINEAR)
                crop_w = rmax - rmin
                ratio = img_patch_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * img_patch_size + np.floor(col_idx * ratio)).astype(np.int64)
                choose = np.array([choose])

                img_masked = norm(img_masked)
                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                index = torch.LongTensor([0])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c = estimator(img_masked, cloud, choose, index)
                try:
                    pred_t, _ = ransac_voting_layer(cloud, pred_t)
                except RuntimeError:
                    print('RANSAC voting fails at No.{0} frame, {1} instance'.format(idx, iid))
                    pred_t = torch.zeros(3).to(pred_r.device)
                    continue
                
                my_t = pred_t.cpu().data.numpy()
                _, which_min = torch.min(pred_c, 1)
                my_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()

                predict_RT = quaternion_matrix(my_r)
                predict_RT[:3, 3] = my_t
                RT_predict.append(predict_RT)

                gt_RT = np.identity(4, np.float32)
                gt_RT[:3, :3] = target_r
                gt_RT[:3, 3] = target_t
                RT_gt.append(gt_RT)
        
        overall_RT_gt[filelist[idx]] = RT_gt
        overall_RT_predict[filelist[idx]] = RT_predict
    
    with open(os.path.join(current_result_dir, 'predict_RT.pkl'), 'wb') as f:
        cPickle.dump(overall_RT_predict, f)
    with open(os.path.join(current_result_dir, 'gt_RT.pkl'), 'wb') as f:
        cPickle.dump(overall_RT_gt, f)

    return overall_RT_gt, overall_RT_predict

def evaluate(obj_name, testing_mode, testing_iter, result_dir, RT_gt_dict, RT_predict_dict):
    # warp the dict to a list
    RT_gt = []
    RT_predict = []
    for k in RT_predict_dict.keys():
        RT_predict += RT_predict_dict[k]
        RT_gt += RT_gt_dict[k]

    dataset_config_dir = os.path.join('./dataset/', obj_name, 'dataset_config')
    cad_model = np.load('{0}/cad.npy'.format(dataset_config_dir))

    if testing_mode == 'virtual':
        experiment_name = testing_mode
    else:
        experiment_name = testing_mode + '_' + str('%02d' % testing_iter)

    current_result_dir = os.path.join(result_dir, obj_name, experiment_name)
    if not os.path.exists(current_result_dir):
        os.makedirs(current_result_dir)

    max_v = np.max(cad_model, axis=0)
    min_v = np.min(cad_model, axis=0)
    object_size = max_v - min_v
    object_diameter = np.linalg.norm(object_size)
    
    # compute add / add-s
    model_pts = cad_model.transpose()
    add = batch_compute_add(np.array(RT_predict), np.array(RT_gt), model_pts)
    
    recall_add = []
    for i in range(100):
        add_threshold = (i + 1) * 0.01 * object_diameter
        correct_num = np.sum(np.array(add) < add_threshold)
        recall_add.append(correct_num / len(add))
    auc_add = compute_auc(add, 0.10 * object_diameter)
    
    evaluation_result_path = current_result_dir + '/result.txt'
    with open(evaluation_result_path, 'w') as f:
        f.write('object diameter: ' + str(object_diameter) + '\n')
        f.write('recall of add: ' + str(recall_add[9]) + '\n')
        f.write('auc of add: ' + str(auc_add) + '\n')
        f.close()
    np.save(os.path.join(current_result_dir, 'add.npy'), np.array(add))

if __name__ == '__main__':
    RT_gt, RT_predict = inference(opt.obj_name, opt.testing_mode, opt.testing_iter, opt.result_dir)
    evaluate(opt.obj_name, opt.testing_mode, opt.testing_iter, opt.result_dir, RT_gt, RT_predict)
