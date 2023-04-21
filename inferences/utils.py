
import mmcv 
import numpy as np
import mmengine

from mmseg.apis import inference_model

import slidingwindow as sw

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

    if color_mask is not None:
        
        img_result[mask_output_bool, :] = img_result[mask_output_bool,:] * (1-alpha) + color_mask * alpha
    

    return img_result, mask_output