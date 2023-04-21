import os 

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import mmcv
import mmengine
import numpy as np
import slidingwindow as sw
from mmseg.apis import init_model, inference_model

import sys
sys.path.append('..')

from datasets import *

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


def inference_segmentor_sliding_window(model, input_img, color_mask, score_thr = 0.1, window_size = 1024, overlap_ratio = 0.5, alpha=0.6):

    """
    Inference by sliding window
    Args:
        model (nn.Module): The loaded detector.
        input_img (str or ndarray): The image filename or loaded image.
        color_mask (ndarray): The color mask for each class.
        score_thr (float): The threshold of bbox score.
        window_size (int): The size of sliding window.
        overlap_ratio (float): The overlap ratio of sliding window.
        alpha (float): The transparency of mask.

    Returns:
        img_result (ndarray): The result image. The shape is (H, W, 3).
        mask_output (ndarray): The result mask. The shape is (H, W).
    """

    # color mask has to be updated for multiple-class object detection
    if isinstance(input_img, str) :
        img = mmcv.imread(input_img)
    else :
        img = input_img

    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
    mask_output = np.zeros((img.shape[0], img.shape[1]), dtype=bool)


    for window in mmengine.track_iter_progress(windows):
        # Add print option for sliding window detection
        img_subset = img[window.indices()]
        results = inference_model(model, img_subset)
        mask_output[window.indices()] = results.pred_sem_seg.data.cpu().numpy()

    mask_output = mask_output.astype(np.uint8)
    mask_output[mask_output > 1] = 1

    mask_output_bool = mask_output.astype(bool)

    # Add colors to detection result on img
    img_result = img
    img_result[mask_output_bool, :] = img_result[mask_output_bool,:] * (1-alpha) + color_mask * alpha

    return img_result, mask_output


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    # get palette 
    color_mask = np.array([[0, 0, 255]])
    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))

    for img_path in mmengine.track_iter_progress(img_list):

        img_result, mask_output = inference_segmentor_sliding_window(model, img_path, color_mask, score_thr = 0.1, window_size = 3096, overlap_ratio=0.1,)
        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        os.makedirs(args.rst_dir, exist_ok=True)

        mmcv.imwrite(img_result, rst_path)
        mmcv.imwrite(mask_output, mask_path)

if __name__ == '__main__':
    main()