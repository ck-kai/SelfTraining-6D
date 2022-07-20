# test 19 NIPS DIB-Renderer
# render multi objects in batch, one in one image
import os.path as osp
import os
import sys
import cv2
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "./"))
from lib.renderer.core.dr_utils.dib_renderer_x import DIBRenderer
from lib.renderer.core.dr_utils.dr_utils import load_objs, render_dib_vc_batch

def heatmap(input, min=None, max=None, to_255=False, to_rgb=False):
    """ Returns a BGR heatmap representation """
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min + 0.001))

    final = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
    if to_rgb:
        final = final[:, :, [2, 1, 0]]
    if to_255:
        return final.astype(np.uint8)
    else:
        return final.astype(np.float32) / 255.0

def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    if titles is not None:
        assert len(ims) == len(titles), "{} != {}".format(len(ims), len(titles))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(ims):
                break
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1

    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            # mmcv.mkdir_or_exist(osp.dirname(save_path))
            plt.savefig(save_path)
    return fig

def rendering(obj_id, pose, intrinsic, height, width, renderer, ren_model):
    models = ren_model

    tensor_args = {"device": "cuda", "dtype": torch.float32}
    Rs = [pose[:3,:3]]
    Ts = [pose[:3, 3]]
    obj_ids = [0 for _ in Rs]
    Ks = [intrinsic for _ in Rs]

    Rs = torch.tensor(Rs).to(**tensor_args)
    Ts = torch.tensor(Ts).to(**tensor_args)

    ren_ims, ren_probs, ren_masks, ren_depths = render_dib_vc_batch(
        renderer, Rs, Ts, Ks, obj_ids, models, rot_type="mat", H=height, W=width, near=0.01, far=100.0, with_depth=True
    )

    rgb_img = ren_ims[0].detach().cpu().numpy()
    prob_map = ren_probs[0, :, :, 0].detach().cpu().numpy()
    mask_img = ren_masks[0, :, :, 0].detach().cpu().numpy()
    depth_map = ren_depths[0].detach().cpu().numpy()

    return rgb_img, prob_map, mask_img, depth_map

def croped_rendering(obj_id, pose, intrinsic, height, width, center_x, center_y, croped_height, croped_width, renderer, ren_model):
    models = ren_model

    # adjust the intrinsics for croped image
    croped_intrinsic = copy.deepcopy(intrinsic)
    croped_intrinsic[0, 2] = intrinsic[0, 2] + float(croped_width - 1) / 2 - center_x
    croped_intrinsic[1, 2] = intrinsic[1, 2] + float(croped_height - 1) / 2 - center_y

    tensor_args = {"device": "cuda", "dtype": torch.float32}
    Rs = [pose[:3,:3]]
    Ts = [pose[:3, 3]]
    obj_ids = [0 for _ in Rs]
    Ks = [croped_intrinsic for _ in Rs]

    Rs = torch.tensor(Rs).to(**tensor_args)
    Ts = torch.tensor(Ts).to(**tensor_args)

    ren_ims, ren_probs, ren_masks, ren_depths = render_dib_vc_batch(
        renderer, Rs, Ts, Ks, obj_ids, models, rot_type="mat", H=croped_height, W=croped_width, near=0.01, far=1.0, with_depth=True
    )

    rgb_img = ren_ims[0].detach().cpu().numpy()
    prob_map = ren_probs[0, :, :, 0].detach().cpu().numpy()
    mask_img = ren_masks[0, :, :, 0].detach().cpu().numpy()
    depth_map = ren_depths[0].detach().cpu().numpy()

    # return rgb_img, prob_map, mask_img, depth_map
    # warp the croped rendering result to the original size
    original_img = np.zeros((height, width, rgb_img.shape[-1]), dtype = rgb_img.dtype)
    original_prob_map = np.zeros((height, width), dtype = prob_map.dtype)
    original_mask = np.zeros((height, width), dtype = mask_img.dtype)
    original_depth = np.zeros((height, width), dtype = depth_map.dtype)

    x1 = max(int(center_x - float(croped_width - 1) / 2), 0)
    x2 = min(x1 + croped_width, width)
    y1 = max(int(center_y - float(croped_height - 1) / 2), 0)
    y2 = min(y1 + croped_height, height)

    x3 = max(int(float(croped_width - 1) / 2 - center_x), 0)
    x4 = x3 + (x2 - x1)
    y3 = max(int(float(croped_height - 1) / 2 - center_y), 0)
    y4 = y3 + (y2 - y1)

    original_img[y1:y2, x1:x2, :] = rgb_img[y3:y4, x3:x4, :]
    original_prob_map[y1:y2, x1:x2] = prob_map[y3:y4, x3:x4]
    original_mask[y1:y2, x1:x2] = mask_img[y3:y4, x3:x4]
    original_depth[y1:y2, x1:x2] = depth_map[y3:y4, x3:x4]

    return original_img, original_prob_map, original_mask, original_depth
