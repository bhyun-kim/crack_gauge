import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import slidingwindow as sw
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
from mmengine import track_progress

"""
The presented script is intended for facilitating multi-scale 
inference in mmsegmentation. It is crucial to note that detecting 
cracks is a task that demands a high degree of precision. As a 
measure to enhance the accuracy of detecting crack images, we perform 
inference using high-resolution images. Conversely, for all other 
classes, we perform inference using images with normal resolutions.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Inference detector')
    parser.add_argument('--crack_config', help='the config file to inference crack')
    parser.add_argument('--crack_checkpoint', help='the checkpoint file to inference crack')
    parser.add_argument('--others_config', help='the config file to inference other classes')
    parser.add_argument('--others_checkpoint', help='the checkpoint file to inference other classes')
    parser.add_argument('--srx_dir', help='the dir to inference')
    parser.add_argument('--rst_dir', help='the dir to save result')
    parser.add_argument('--srx_suffix', default='.png', help='the source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='the result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='the mask output extension')
    parser.add_argument('--alpha', default=0.8, help='the alpha value for blending')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')
    others_model = init_model(args.others_config, args.others_checkpoint, device='cuda:1')

    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))

    palette = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 255]]

    for img_path in img_list:

        crack_result = inference_model(crack_model, img_path)
        others_result = inference_model(others_model, img_path)

        mask_result = others_result.pred_sem_seg.data.cpu().numpy()
        mask_result = np.squeeze(mask_result)

        mask_result[mask_result > 0] += 1

        crack_mask = crack_result.pred_sem_seg.data.cpu().numpy()
        crack_mask = np.squeeze(crack_mask)

        mask_result[crack_mask == 1] = 1

        vis_result = mmcv.imread(img_path)
    
        for i in range(1, 2):
            color = palette[i]
            color = np.array(color, dtype=np.uint8)
            mask_bool = mask_result == i

            vis_result[mask_bool, :] = vis_result[mask_bool, :]*(1-args.alpha)+ color*args.alpha
        
        for i in range(2, 5):
            color = palette[i]
            color = np.array(color, dtype=np.uint8)
            mask_bool = mask_result == i

            vis_result[mask_bool, :] = vis_result[mask_bool, :]*(1-args.alpha)+ color*args.alpha
            

        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        mask_result = np.squeeze(mask_result)
        
        mmcv.imwrite(vis_result, rst_path)
        mmcv.imwrite(mask_result, mask_path)


if __name__ == '__main__':
    main()








