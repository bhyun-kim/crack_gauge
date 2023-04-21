import os 

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import mmcv
import mmengine
import numpy as np
from mmseg.apis import init_model, inference_model

from skimage.morphology import medial_axis
from skimage.measure import label, regionprops_table

import cv2

import sys
sys.path.append('..')

from datasets import *

from quantify_seg_results import quantify_crack_width_length
from utils import inference_segmentor_sliding_window

"""
Classes of Concrete Damage Dataset

0. Background -> No quantification needed 
1. Crack -> Length and Width

"""


def parse_args():
    parser = argparse.ArgumentParser(description='Inference detector')
    parser.add_argument('--config', help='the config file to inference')
    parser.add_argument('--checkpoint', help='the checkpoint file to inference')
    parser.add_argument('--srx_dir', help='the dir to inference')
    parser.add_argument('--rst_dir', help='the dir to save result')
    parser.add_argument('--srx_suffix', default='.png', help='the source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='the result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='the mask output extension')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    # get palette 
    color_mask = np.array([[0, 0, 255]])
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))

    for img_path in mmengine.track_iter_progress(img_list):

        seg_result, mask_output = inference_segmentor_sliding_window(model, img_path, color_mask, score_thr = 0.1, window_size = 3096, overlap_ratio=0.1,)
        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        seg_result = quantify_crack_width_length(mask_output, seg_result, (0, 0, 255))

        os.makedirs(args.rst_dir, exist_ok=True)

        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(mask_output, mask_path)


if __name__ == '__main__':
    main()