from mmdet import registry
from mmdet.datasets.coco import CocoDataset


@registry.DATASETS.register_module()
class ConcreteCrackCOCODataset(CocoDataset):
    """
    Concrete Crack COCO Dataset

    !!!CAUTION!!!
    The palette is BGR, not RGB
    """

    METAINFO = {
        'classes': ('crack', ),
        'palette': [(0, 0, 255)]
    }