import numpy as np

import mmcv

from mmseg.datasets.cityscapes import CityscapesDataset
from mmseg.registry import DATASETS

from mmengine import track_iter_progress, FileClient

@DATASETS.register_module()
class ConcreteDamageDataset(CityscapesDataset):
    """Concrete Damage Dataset using Cityscaps Dataset Format
    
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelIds.png' for Concrete Damage Dataset. 
    """
    METAINFO = dict(
        classes=('background', 'efflorescence', 'rebar_exposure', 'spalling', 'corrosion'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255]]
    )

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 label_map=None,
                 **kwargs):
        super(CityscapesDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

        if label_map :
            self.serialize_data = False
            self.label_map = label_map
            self._metainfo['label_map'] = self.label_map
            self.data_list = self.load_data_list()
        
    def get_cat_ids(self, idx):
        """Get category ids of the idx-th data sample.
        Args:
            idx (int): Index of data.
        
        Returns:
            cat_ids (list[int]): Category ids of the idx-th data sample.
        """
        if hasattr(self, 'cat_ids'):
            pass
        else:
            self.get_gt_statistics()
            
        old_cat_ids = self.cat_ids[idx]
        cat_ids = []

        for old_id, new_id in self._metainfo['label_map'].items():
            if old_id in old_cat_ids:
                cat_ids.append(new_id)

        return cat_ids

    def get_gt_statistics(self):
        """Get statistics of ground truth.
        """

        self.cat_ids = []        
        fileclient = FileClient.infer_client(dict(backend='disk'))

        for idx in track_iter_progress(range(len(self))):

            data = self.get_data_info(idx)

            segmap_bytes = fileclient.get(data['seg_map_path'])
            seg_map = mmcv.imfrombytes(
            segmap_bytes, flag='unchanged').squeeze().astype(np.uint8)

            self.cat_ids.append(np.unique(seg_map).tolist())

