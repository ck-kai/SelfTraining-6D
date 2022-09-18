from auto_label import *
import os
import sys
import glob
import argparse
import random
import time
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from lib.network import PoseNet
from lib.loss import Loss
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.utils import setup_logger
from dataset.zigzag.dataset import PoseDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='zigzag')
parser.add_argument('--renderer_model_dir', type=str, default='renderer/robi_models')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--num_rot', type=int, default=60, help='number of rotation anchors')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--noise_trans', default=5.00, help='random noise added to translation')
parser.add_argument('--lr', default=0.00001, help='learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_posenet', type=str, default='pose_model.pth', help='resume PoseNet model')
parser.add_argument('--nepoch', type=int, default=10, help='max number of epochs to train')
parser.add_argument('--iter_start', type=int, default=0, help='the start number of iterations for self-training')
parser.add_argument('--best_metric', type=float, default=100000.0)
parser.add_argument('--iter', type=int, default=5, help='the total number of iterations for self-training')
parser.add_argument('--validation', action='store_true', default=False, help='valid the real model with a small number of real data')
opt = parser.parse_args()

opt.num_objects = 1 #number of object classes in the dataset
opt.num_points = 1000 #number of points on the input pointcloud
opt.num_rot = 60

intrinsics = np.identity(3, np.float32)
cam_cx = 379.32687
cam_cy = 509.43720
cam_fx = 1083.09705
cam_fy = 1083.09705
intrinsics[0, 0] = cam_fx
intrinsics[1, 1] = cam_fy
intrinsics[0, 2] = cam_cx
intrinsics[1, 2] = cam_cy

# function used to update the train list for real data after each time of label/re-label
def update_train_test_split(data_path, config_dir):
    mask_pathes = glob.glob(os.path.join(data_path, '*_mask.png'))
    mask_pathes = sorted(mask_pathes, key=lambda a:a.split('/')[-1].split('.')[0])

    train_list_path = os.path.join(config_dir, 'st_real_train_list.txt')
    f = open(train_list_path, 'w')
    for i in range(len(mask_pathes)):
        f.write(mask_pathes[i].split('/')[-1].split('.')[0][:-5] + '\n')
    f.close()

import logging
def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l

def self_training_with_real_data(iter_idx = 1, estimator_best_test = np.Inf, robi_object = '01'):
    opt.model_dir = 'real_models/' + robi_object + '/' + str(iter_idx).zfill(2) #folder to save trained models
    opt.log_dir = opt.model_dir + '/logs'
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    print("init model")
    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot = opt.num_rot)
    print("model initialized")
    estimator.cuda()

    print("load model")
    opt.resume_dir = 'real_models/' + robi_object + '/best'
    estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.resume_dir, opt.resume_posenet)))

    print("init dataset")
    opt.train_dataset_root = './data/' + robi_object + '/teacher_label_iter_' + str(iter_idx).zfill(2)
    dataset = PoseDataset('train', opt.num_points, True, opt.train_dataset_root, opt.noise_trans)
    print("dataset root:",opt.train_dataset_root,"#samples:",len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    opt.diameters = dataset.get_diameter()
    criterion = Loss(opt.sym_list, estimator.rot_anchors)
    k_nearest = 1

    optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
    global_step = (len(dataset) // opt.batch_size) * (opt.start_epoch - 1)
    st_time = time.time()
    best_test = estimator_best_test
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('{0}_estimator_{1}'.format(iter_idx, epoch), os.path.join(opt.log_dir, 'finetune_estimator_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_loss_avg = 0.0
        train_loss_r_avg = 0.0
        train_loss_t_avg = 0.0
        train_loss_reg_avg = 0.0
        estimator.train()
        optimizer.zero_grad()
        for i, data in enumerate(dataloader, 0):
            points, choose, img, target_t, target_r, model_points, idx, gt_t = data
            obj_diameter = opt.diameters[idx]
            points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                            Variable(choose).cuda(), \
                                                                            Variable(img).cuda(), \
                                                                            Variable(target_t).cuda(), \
                                                                            Variable(target_r).cuda(), \
                                                                            Variable(model_points).cuda(), \
                                                                            Variable(idx).cuda()
            pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
            
            loss, loss_r, loss_t, loss_reg = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
            loss.backward()
            train_loss_avg += loss.item()
            train_loss_r_avg += loss_r.item()
            train_loss_t_avg += loss_t.item()
            train_loss_reg_avg += loss_reg.item()
            train_count += 1
            if train_count % opt.batch_size == 0:
                global_step += 1
                lr = opt.lr
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4:f}'.format(time.strftime("%Hh %Mm %Ss", 
                    time.gmtime(time.time()-st_time)), epoch, int(train_count/opt.batch_size), train_count, train_loss_avg/opt.batch_size))
                train_loss_avg = 0.0
                train_loss_r_avg = 0.0
                train_loss_t_avg = 0.0
                train_loss_reg_avg = 0.0

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        if opt.validation:
            opt.val_dataset_root = './data/' + robi_object + '/validation_data'
            val_dataset = PoseDataset('test', opt.num_points, False, opt.val_dataset_root, 0.0)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

            logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Validation started'))
            test_dis = 0.0
            test_count = 0
            estimator.eval()
            for j, data in enumerate(val_dataloader, 0):
                points, choose, img, target_t, target_r, model_points, idx, gt_t = data
                obj_diameter = opt.diameters[idx]
                points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                            Variable(choose).cuda(), \
                                                                            Variable(img).cuda(), \
                                                                            Variable(target_t).cuda(), \
                                                                            Variable(target_r).cuda(), \
                                                                            Variable(model_points).cuda(), \
                                                                            Variable(idx).cuda()
                with torch.no_grad():
                    pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
                loss, _, _, _ = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
                test_count += 1

                # evalaution
                how_min, which_min = torch.min(pred_c, 1)
                pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
                pred_r = quaternion_matrix(pred_r)[:3, :3]
                try:
                    pred_t, pred_mask = ransac_voting_layer(points, pred_t)
                except RuntimeError:
                    print('RANSAC voting fails')
                    continue

                pred_t = pred_t.cpu().data.numpy()
                model_points = model_points[0].cpu().detach().numpy()
                pred = np.dot(model_points, pred_r.T) + pred_t
                target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
                if idx[0].item() in opt.sym_list:
                    pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                    target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                    inds = KNearestNeighbor.apply(target.unsqueeze(0), pred.unsqueeze(0),k_nearest)
                    target = torch.index_select(target, 1, inds.view(-1) - 1)
                    dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
                else:
                    dis = np.mean(np.linalg.norm(pred - target, axis=1))
                logger.info('Test time {0} Test Frame No.{1} loss:{2:f} confidence:{3:f} distance:{4:f}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss, how_min[0].item(), dis))
                test_dis += dis
            
            test_dis = test_dis / test_count
            logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
            if test_dis < best_test:
                best_test = test_dis
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.model_dir, epoch, test_dis))
                torch.save(estimator.state_dict(), '{0}/pose_model.pth'.format(opt.model_dir))
                torch.save(estimator.state_dict(), '{0}/pose_model.pth'.format(opt.resume_dir))
                logger.info('%d >>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<' % epoch)
        else: # if there is no validation data, directly save model of the last epoch
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}.pth'.format(opt.model_dir, epoch))
            torch.save(estimator.state_dict(), '{0}/pose_model.pth'.format(opt.model_dir))
            torch.save(estimator.state_dict(), '{0}/pose_model.pth'.format(opt.resume_dir))
    return best_test

def iterative_self_training(robi_object, start_iterations, num_iterations):
    estimator_best_test = opt.best_metric

    for iter in range(start_iterations, num_iterations):
        if iter == 0:
            resume_model_dir = './real_models/' + robi_object + '/best'
            if not os.path.exists(resume_model_dir):
                os.makedirs(resume_model_dir)
            virtual_model_path = './virtual_models/' + robi_object + '/pose_model.pth'
            if not os.path.exists(virtual_model_path):
                print('error, the initial model does not exist!')
                sys.exit()
            shutil.copy(virtual_model_path, resume_model_dir)

        # # delete the old training data to save the storage
        # if iter >= 3:
        #     old_training_data_dir = os.path.join('./data', robi_object, 'teacher_label_iter_' + str(iter-1).zfill(2))
        #     shutil.rmtree(old_training_data_dir)

        print("label_poses_with_teacher...")
        label_poses_with_teacher(iter + 1, renderer_model_dir = opt.renderer_model_dir, obj_id = robi_object, intrinsics=intrinsics)
        print("update_train_test_split...")
        update_train_test_split('./data/' + robi_object + '/teacher_label_iter_' + str(iter+1).zfill(2), './dataset/' + robi_object +'/dataset_config')
        print("self_training_with_real_data...")
        estimator_best_test = self_training_with_real_data(iter + 1, estimator_best_test, robi_object)

if __name__ == '__main__':
    print('Self-training for {0}, and will start training from {1}-th iterations and will stop at {2}-th iterations'.format(opt.dataset, opt.iter_start+1, opt.iter))
    print('Initial best metric valus is {0}'.format(opt.best_metric))
    iterative_self_training(opt.dataset, opt.iter_start, opt.iter)
