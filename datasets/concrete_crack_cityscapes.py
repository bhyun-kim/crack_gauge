from mmseg import registry
from .concrete_damage_cityscapes import ConcreteDamageDataset

@registry.DATASETS.register_module()
class CrackCityscapesDataset(ConcreteDamageDataset):
    """Concrete Crack Dataset using Cityscaps Dataset Format
    
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelIds.png' for Concrete Damage Dataset. 

    !!!CAUTION!!!
    The palette is BGR, not RGB
    """
    METAINFO = dict(
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [0, 0, 255]]
    )