from mmseg import registry
from .uos_concrete_damage import UOSConcreteDamageDataset


@registry.DATASETS.register_module()
class UOSCrackDataset(UOSConcreteDamageDataset):
    """University of Concrete Crack Dataset
    This dataset follows the same format as Cityscapes.
    
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelIds.png' for Concrete Damage Dataset. 

    Note that palette is BGR.
    """
    METAINFO = dict(classes=('background', 'crack'),
                    palette=[[0, 0, 0], [0, 0, 255]])
