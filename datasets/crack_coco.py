from mmdet import registry
from mmdet.datasets.coco import CocoDataset


@registry.DATASETS.register_module()
class ConcreteCrackCOCODataset(CocoDataset):
    """
    Concrete Crack COCO Dataset
    
    Note that palette is BGR.
    """

    METAINFO = {'classes': ('crack', ), 'palette': [(0, 0, 255)]}
