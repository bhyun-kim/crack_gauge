import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import argparse
from glob import glob

from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
from mmengine import track_progress

from quantify_seg_results import quantify_crack_width_length, quantify_deterioration_area

from torch.cuda import empty_cache
from utils import inference_segmentor_sliding_window

"""
The presented script is intended for facilitating multi-scale 
inference in mmsegmentation. It is crucial to note that detecting 
cracks is a task that demands a high degree of precision. As a 
measure to enhance the accuracy of detecting crack images, we perform 
inference using high-resolution images. Conversely, for all other 
classes, we perform inference using images with normal resolutions. All the other classes are called deterio which stands for deterioration here.

The crack model should only have 'background' and 'crack' classes.
The deterioration model should have all the classes except 'crack'.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Inference detector')
    parser.add_argument('--crack_config', help='the config file to inference crack')
    parser.add_argument('--crack_checkpoint', help='the checkpoint file to inference crack')
    parser.add_argument('--deterio_config', help='the config file to inference other classes')
    parser.add_argument('--deterio_checkpoint', help='the checkpoint file to inference other classes')
    parser.add_argument('--srx_dir', help='the dir to inference')
    parser.add_argument('--rst_dir', help='the dir to save result')
    parser.add_argument('--srx_suffix', default='.png', help='the source image extension')
    parser.add_argument('--rst_suffix', default='.png', help='the result image extension')
    parser.add_argument('--mask_suffix', default='.png', help='the mask output extension')
    parser.add_argument('--alpha', default=0.8, help='the alpha value for blending')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='convert rgb to bgr, if the model palette is written in rgb format.')
    parser.set_defaults(rgb_to_bgr=False)
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='overwrite the crack palette with black and red. To be used when the crack model is trained with a different palette.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # TO-DO: manage GPU memory to avoid OOM
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda:0')
    deterio_model = init_model(args.deterio_config, args.deterio_checkpoint, device='cuda:0')

    if 'crack' in deterio_model.dataset_meta['classes']:
        raise ValueError('The deterioration model should not have crack class.')

    img_list = glob(os.path.join(args.srx_dir, f'*{args.srx_suffix}'))

    crack_palette=crack_model.dataset_meta['palette'][:2] # only keep the first two colors, we usally assign the first color to background and the second color to crack
    if args.overwrite_crack_palette:
        crack_palette = [[0, 0, 0], [0, 0, 255]] # we use black and white to represent crack and background
    deterio_palette=deterio_model.dataset_meta['palette']
    palette = crack_palette + deterio_palette[1:]

    if args.rgb_to_bgr:
        palette = [p[::-1] for p in palette]

    classes = crack_model.dataset_meta['classes'] + deterio_model.dataset_meta['classes'][1:]

    for img_path in img_list:
        
        _, crack_mask = inference_segmentor_sliding_window(crack_model, img_path, color_mask=None, score_thr = 0.1, window_size = 2048, overlap_ratio=0.1)

        deterio_seg_result = inference_model(deterio_model, img_path)

        mask_result = deterio_seg_result.pred_sem_seg.data.cpu().numpy()
        mask_result = np.squeeze(mask_result)

        # increase the deteroration mask by 1, so that the crack mask can be added to it. 
        mask_result[mask_result > 0] += 1

        mask_result[crack_mask == 1] = 1

        seg_result = mmcv.imread(img_path)

        # visualize the crack mask
        color = palette[1]
        color = np.array(color, dtype=np.uint8)
        mask_bool = crack_mask == 1

        seg_result[mask_bool, :] = seg_result[mask_bool, :]*(1-args.alpha)+ color*args.alpha
        
        # visualize the deterioration mask
        for i in range(2, len(palette)):
            color = palette[i]
            color = np.array(color, dtype=np.uint8)
            mask_bool = mask_result == i

            seg_result[mask_bool, :] = seg_result[mask_bool, :]*(1-args.alpha)+ color*args.alpha
            
        rst_name = os.path.basename(img_path).replace(args.srx_suffix, args.rst_suffix)
        mask_name = os.path.basename(img_path).replace(args.srx_suffix, args.mask_suffix)

        # quantify crack 

        seg_result = quantify_crack_width_length(seg_result, crack_mask, palette[1])
        # create vis_config 
        vis_config = []

        for i in range(2, len(palette)):
            vis_config.append([i, classes[i], palette[i]])

        seg_result = quantify_deterioration_area(seg_result, mask_result, vis_config)

        rst_path = os.path.join(args.rst_dir, rst_name)
        mask_path = os.path.join(args.rst_dir, mask_name)

        mask_result = np.squeeze(mask_result)
        
        mmcv.imwrite(seg_result, rst_path)
        mmcv.imwrite(mask_result, mask_path)


if __name__ == '__main__':
    main()