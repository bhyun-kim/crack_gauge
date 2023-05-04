import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

import slidingwindow as sw
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
from mmengine import track_progress


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

    model = init_model(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))
    
    model.dataset_meta['palette'] = [[0, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 165, 255]]

    for img_path in img_list :# track_progress(img_list):

        result = inference_model(model, img_path)

        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        # vis_result = show_result_pyplot(model, img_path, result, show=False)


        mask_result = result.pred_sem_seg.data.cpu().numpy()
        mask_result = np.squeeze(mask_result)

        vis_result = mmcv.imread(img_path)
        for i in range(1, 2):
            color = model.dataset_meta['palette'][i]
            vis_result[mask_result == i] = vis_result[mask_result == i] * 0.5 + np.array(color) * 0.5
        
        mmcv.imwrite(vis_result, rst_path)
        mmcv.imwrite(mask_result, mask_path)


if __name__ == '__main__':
    main()








