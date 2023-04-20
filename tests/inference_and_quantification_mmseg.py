import os 

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import mmcv
import mmengine
import numpy as np
import slidingwindow as sw
from mmseg.apis import init_model, inference_model

from skimage.morphology import medial_axis
from skimage.measure import label, regionprops_table

import cv2

import sys
sys.path.append('..')

from datasets import *


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


def connect_cracks(mask_output, epsilon = 50):
    """
    Connect cracks by using medial axis

    Args:
        mask_output (ndarray): The result mask. The shape is (H, W).

    Returns:
        mask_output (ndarray): The result mask. The shape is (H, W).
    """

    # label each crack
    labels, num = label(mask_output, connectivity=2, return_num=True)
    # get information of each crack area
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords', 'orientation'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']

    crack_region_table['is_horizontal'] = width > height

    connecting_directions = ['x_axis', 'y_axis']
    connect_line_img = np.zeros_like(mask_output, dtype=np.uint8)

    for connecting_direction in connecting_directions:

        e2_list = []
        e1_list = []

        for crack_num, crack_region in enumerate(crack_region_table['label']):

            min_row = crack_region_table['bbox-0'][crack_num]
            min_col = crack_region_table['bbox-1'][crack_num]
            max_row = crack_region_table['bbox-2'][crack_num] - 1
            max_col = crack_region_table['bbox-3'][crack_num] - 1

            if crack_region_table['is_horizontal'][crack_num]:
                # max col / min col
                col = crack_region_table['coords'][crack_num][:, 1]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(col == max_col), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(col == min_col), :][0][0]

                if connecting_direction == 'y_axis' and e2[0] < e1[0]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

            else:
                # max row / min row
                row = crack_region_table['coords'][crack_num][:, 0]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(row == max_row), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(row == min_row), :][0][0]

                if connecting_direction == 'x_axis' and e2[1] < e1[1]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

        crack_region_table['e2'] = e2_list
        crack_region_table['e1'] = e1_list


        n = len(crack_region_table['label'])
        color = (1)  # binary image


        for num_e2, e2 in enumerate(crack_region_table['e2']):

            connect_candidates_e2 = []
            connect_candidates_e1 = []
            distance_list = []

            for num_e1, e1 in enumerate(crack_region_table['e1']):

                if num_e2 != num_e1:
                    d = np.subtract(e1, e2)
                    distance = np.sqrt(d[0] ** 2 + d[1] ** 2)

                    if (distance < epsilon):
                        distance_list.append(distance)
                        connect_candidates_e2.append(tuple(e2[::-1]))
                        connect_candidates_e1.append(tuple(e1[::-1]))

            if distance_list :
                connect_idx = np.argmin(distance_list)
                connect_e2 = connect_candidates_e2[connect_idx]
                connect_e1 = connect_candidates_e1[connect_idx]
                connect_line_img = cv2.line(connect_line_img, connect_e2, connect_e1, color, 8)

    mask_output = mask_output + connect_line_img
    mask_output[mask_output > 1] = 1

    return mask_output

def create_distance_map(mask):
    """
    Create distance map from mask

    Args:
        mask (ndarray): The mask image. The shape is (H, W).

    Returns:
        distance_map (ndarray): The distance map. The shape is (H, W).
    """

    dist, skel = medial_axis(mask, return_distance=True)
    distance_map = dist * skel

    return distance_map


def quantify_crack_width_length(crack_mask, distance_map):
    """
    Quantify crack width and length

    Args: 
        crack_mask (ndarray): The crack mask image. The shape is (H, W).
        distance_map (ndarray): The distance map. The shape is (H, W).

    Returns:
        crack_width (float): The crack width.
        crack_length (float): The crack length.
    """
    crack_distance_map = distance_map * crack_mask

    crack_width = np.mean(crack_distance_map[crack_distance_map > 0])

    crack_length = np.sum(crack_distance_map > 0)

    return crack_width, crack_length

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

        # create distance map
        distance_map = create_distance_map(mask_output)
                
        # label mask 
        mask_output = connect_cracks(mask_output)
        mask_label = label(mask_output)
        # regionprops_table
        crack_region_table = regionprops_table(mask_label)

        # loop through each crack
        for crack_id in np.unique(mask_label)[1:]:
            crack_mask = mask_label == crack_id

            if np.sum(crack_mask) < 500:
                continue

            crack_width, crack_length = quantify_crack_width_length(crack_mask, distance_map)

            # get crack x, y   
            crack_num = crack_id - 1

            minr = crack_region_table['bbox-0'][crack_num]
            minc = crack_region_table['bbox-1'][crack_num]
            maxr = crack_region_table['bbox-2'][crack_num]
            maxc = crack_region_table['bbox-3'][crack_num]

            # clip minr, minc, maxr, maxc
            textr = max(minr, 20)
            textc = max(minc, 20)

            # display on image
            img_result = cv2.putText(
                img_result, f'Crack: {crack_width:.2f} x {crack_length:.2f}', (textc, textr), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # put rectangle on crack
            img_result = cv2.rectangle(img_result, (minc, minr), (maxc, maxr), (0, 0, 255), 1)

        os.makedirs(args.rst_dir, exist_ok=True)

        mmcv.imwrite(img_result, rst_path)
        mmcv.imwrite(mask_output, mask_path)


if __name__ == '__main__':
    main()