import numpy as np 
import cv2

from skimage.measure import label, regionprops_table
from skimage.morphology import medial_axis

def connect_cracks_by_edge(mask_output, epsilon = 50):
    """
    Connect the edges of adjacent cracks

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


def _calculate_crack_width_length(crack_mask, distance_map):
    """
    Calculate crack width and length

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



def quantify_crack_width_length(seg_result, mask_output, color, minimum_area=500, line_thickness=2):
    """
    Quantify crack width and length. The word 'quantify' means to calculate the crack width and length and visualize them one the segmentation result image. 
    
    Args:
        seg_result (ndarray): The segmentation result. The shape is (H, W, C).
        mask_output (ndarray): The crack mask image. The shape is (H, W).
        color (tuple): The color of the crack width and length. The shape is (3,).
        minimum_area (int): The minimum crack area. The default value is 500.
        line_thickness (int): The thickness of the crack width and length. The default value is 2.
        
    Returns:
        seg_result (ndarray): 
    """

    # determine font scale and line thickness of text
    font_scale = seg_result.shape[0] / 1000
    font_thickness = int(line_thickness * font_scale)

    # create distance map
    distance_map = create_distance_map(mask_output)
            
    # label mask 
    mask_output = connect_cracks_by_edge(mask_output)
    mask_label = label(mask_output)
    # regionprops_table
    crack_region_table = regionprops_table(mask_label)

    # loop through each crack
    for crack_id in np.unique(mask_label)[1:]:
        crack_mask = mask_label == crack_id

        if np.sum(crack_mask) < minimum_area:
            continue

        crack_width, crack_length = _calculate_crack_width_length(crack_mask, distance_map)

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
        seg_result = cv2.putText(
            seg_result, f'Crack: {crack_width:.2f} x {crack_length:.2f}', (textc, textr), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
        
        # put rectangle on crack
        seg_result = cv2.rectangle(seg_result, (minc, minr), (maxc, maxr), color, line_thickness)

    return seg_result


def check_vis_config(vis_config):
    """
    Check visualization configuration for the function "quantify_deterioration_area".

    Args: 
        vis_config (list): Visualization configuration. It should be nested list includes [class_idx (int), class_name (str), color (tuple)]

    Returns:
        None
    """
    assert isinstance(vis_config, list), 'vis_config should be list'

    # assert elements in vis_config
    for config in vis_config:
        assert isinstance(config, list), 'elements in vis_config should be list'
        assert len(config) == 3, 'elements in vis_config should have 3 elements including [class_idx (int), class_name (str), color (tuple)]'
        assert isinstance(config[0], int), 'class_idx should be int'
        assert isinstance(config[1], str), 'class_name should be str'
        assert isinstance(config[2], (tuple, list)), 'color should be tuple or list'

    return None


def quantify_deterioration_area(seg_result, seg_mask, vis_config, minimum_area=500, line_thickness=2):
    
    """
    Quantify deterioration area. The word 'quantify' means to calculate the deterioration area and visualize them one the segmentation result image.

    Args: 
        seg_result (ndarray): The segmentation result. The shape is (H, W, C).
        seg_mask (ndarray): The segmentation mask. The shape is (H, W).
        vis_config (list): Visualization configuration. It should be nested list includes [class_idx (int), class_name (str), color (tuple or list)]
    """
    
    assert seg_result.dtype == np.uint8, 'seg_result should be np.uint8'
    check_vis_config(vis_config)

    # determine font scale and line thickness of text
    font_scale = seg_result.shape[0] / 1000
    font_thickness = int(line_thickness * font_scale)

    for config in vis_config:
        class_idx = config[0]
        class_name = config[1]
        color = config[2]

        _seg_mask = seg_mask == class_idx
        
        seg_label = label(_seg_mask.astype(np.uint8))
        seg_region_table = regionprops_table(seg_label)



        for label_id in np.unique(seg_label)[1:]:
            obj_mask = seg_label == label_id

            if np.sum(obj_mask) < minimum_area:
                continue

            # get object width and height 
            obj_num = label_id - 1

            minr = seg_region_table['bbox-0'][obj_num]
            minc = seg_region_table['bbox-1'][obj_num]
            maxr = seg_region_table['bbox-2'][obj_num]
            maxc = seg_region_table['bbox-3'][obj_num]

            obj_width = maxc - minc
            obj_height = maxr - minr

            textc = max(minc, 20)
            textr = max(minr, 20)

            seg_result = cv2.putText(
                seg_result, f'{class_name}:{obj_width:.2f}x{obj_height:.2f}', (textc, textr), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
            
            seg_result = cv2.rectangle(seg_result, (minc, minr), (maxc, maxr), color, line_thickness)
    
    return seg_result