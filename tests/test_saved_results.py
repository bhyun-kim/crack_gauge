import argparse

from glob import glob

import os.path as osp
import numpy as np
import cv2

def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat


def parse_args():
    parser = argparse.ArgumentParser(description='Test saved results')
    parser.add_argument('--pred_dir', help='the dir to save predictions')
    parser.add_argument('--gt_dir', help='the dir to save ground truth')
    parser.add_argument(
        '--pred_suffix', 
        default='.png',
        help='the suffix of prediction files')
    parser.add_argument(
        '--gt_suffix', 
        default='.png',
        help='the suffix of ground truth files')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read img from folder
    pred_list = glob(osp.join(args.pred_dir, '*' + args.pred_suffix)) 

    for pred_path in pred_list:
        
        pred_name = osp.basename(pred_path)
        gt_name = pred_name.replace(args.pred_suffix, args.gt_suffix)
        gt = osp.join(args.gt_dir, gt_name)
        assert osp.exists(gt), f'gt {gt} does not exist'

        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(gt, cv2.IMREAD_UNCHANGED)

        # get confusion matrix
        cm = get_confusion_matrix(pred, gt, 2, 255)
        print(cm)



if __name__ == '__main__':
    main()