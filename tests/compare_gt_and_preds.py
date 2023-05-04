import argparse
import os 

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

from glob import glob

import os.path as osp
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Test saved results')
    parser.add_argument('--pred_dir', help='the dir to save predictions')
    parser.add_argument('--gt_dir', help='the dir to save ground truth')
    parser.add_argument('--srx_img_dir', help='the dir to load images')
    parser.add_argument('--target_img_dir', help='the dir to save images')
    parser.add_argument(
        '--pred_suffix', 
        default='.png',
        help='the suffix of prediction files')
    parser.add_argument(
        '--gt_suffix', 
        default='.png',
        help='the suffix of ground truth files')
    parser.add_argument(
        '--img_suffix',
        default='.jpg',
        help='the suffix of images')
    parser.add_argument(
        '--target_img_suffix',
        default='.jpg',
        help='the suffix of target images'
    )
    
    args = parser.parse_args()
    return args

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

        # dialate the gt to get more accurate results
        gt_dl = cv2.dilate(gt, np.ones((191, 191), np.uint8), iterations=1)

        # get precision 
        tp = np.sum((pred == 1) & (gt_dl == 1))
        fp = np.sum((pred == 1) & (gt_dl == 0))

        precision = tp / (tp + fp)

        # get recall 

        # dilate the pred to get more accurate results
        pred_dl = cv2.dilate(pred, np.ones((191, 191), np.uint8), iterations=1)

        tp = np.sum((pred_dl == 1) & (gt == 1))
        fn = np.sum((pred_dl == 0) & (gt == 1))

        recall = tp / (tp + fn)

        # print image name
        print(pred_name)
        # print precision
        print(f'Precision: {precision * 100:.2f}%')
        # print recall in percentage and under 2 decimal places
        print(f'Recall: {recall * 100:.2f}%')

        # read image from folder
        img_name = pred_name.replace(args.pred_suffix, args.img_suffix)
        img_path = osp.join(args.srx_img_dir, img_name)
        assert osp.exists(img_path), f'img {img_path} does not exist'

        img = cv2.imread(img_path)

        # draw tp and fp on image
        tp = np.where((pred == 1) & (gt_dl == 1))
        # make mask out of tp to dilate it
        tp_mask = np.zeros_like(pred)
        for i in range(len(tp[0])):
            tp_mask[tp[0][i], tp[1][i]] = 1
        # dilate tp to make it more visible
        tp_mask = cv2.dilate(tp_mask, np.ones((100, 100), np.uint8), iterations=1)
        tp = np.where(tp_mask == 1)

        # in brg format   
        for i in range(len(tp[0])):
            # dilate tp to make it more visible
            img[tp[0][i], tp[1][i]] = (0, 0, 255)

        del tp_mask, tp

        fp = np.where((pred == 1) & (gt_dl == 0))
        # make mask out of fp to dilate it
        fp_mask = np.zeros_like(pred)
        for i in range(len(fp[0])):
            fp_mask[fp[0][i], fp[1][i]] = 1
        # dilate fp to make it more visible
        fp_mask = cv2.dilate(fp_mask, np.ones((100, 100), np.uint8), iterations=1)
        fp = np.where(fp_mask == 1)

        for i in range(len(fp[0])):
            # dilate fp to make it more visible
            img[fp[0][i], fp[1][i]] = (0, 255, 0)

        del fp_mask, fp


        # draw fn on image
        fn = np.where((pred_dl == 0) & (gt == 1))
        # make mask out of fn to dilate it
        fn_mask = np.zeros_like(pred)
        for i in range(len(fn[0])):
            fn_mask[fn[0][i], fn[1][i]] = 1
        # dilate fn to make it more visible
        fn_mask = cv2.dilate(fn_mask, np.ones((100, 100), np.uint8), iterations=1)
        fn = np.where(fn_mask == 1)

        for i in range(len(fn[0])):
            # dilate fn to make it more visible
            img[fn[0][i], fn[1][i]] = (255, 0, 0)

        del fn_mask, fn

        # save image
        save_path = osp.join(args.target_img_dir, img_name.replace(args.img_suffix, args.target_img_suffix))
        # resize image to 1/8 of original size
        img = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
        # create folder if not exist
        if not osp.exists(args.target_img_dir):
            os.makedirs(args.target_img_dir)
        cv2.imwrite(save_path, img) 

if __name__ == '__main__':
    main()