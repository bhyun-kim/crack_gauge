from mmdet.datasets.coco import CocoDataset

from mmdet.registry import DATASETS


@DATASETS.register_module()
class ConcreteCrackCOCODataset(CocoDataset):
    """Concrete Crack COCO Dataset"""

    METAINFO = {
        'classes': ('crack', ),
        'palette': [(255, 0, 0)]
    }