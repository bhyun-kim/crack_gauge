from mmdet import registry
from mmdet.datasets.coco import CocoDataset


@registry.DATASETS.register_module()
class ConcreteCrackCOCODataset(CocoDataset):
    """Concrete Crack COCO Dataset"""

    METAINFO = {
        'classes': ('crack', ),
        'palette': [(255, 0, 0)]
    }